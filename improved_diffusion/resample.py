from abc import ABC, abstractmethod
import math
import numpy as np
import torch as th
import torch.distributed as dist
from scipy.stats import multivariate_normal


def create_named_schedule_sampler(name, diffusion, **kwargs):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    elif name == 'fast':
        return FastSampler(diffusion, **kwargs)
    elif name == 'es':
        return ESSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    扩散过程中各个时间步的分布，旨在减少目标的方差。
    默认情况下，抽样器执行无偏重要性抽样，其中目标的均值保持不变。
    然而，子类可以覆盖sample()方法，以改变重新抽样的项如何被重新加权，从而允许实际改变目标。
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        这个函数是一个用于在批次中进行重要性采样（importance sampling）的实现。
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        # 修改的是timestep的采样概率p
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])
        self.name = 'Uniform'

    def weights(self):
        return self._weights


class FastSampler(ScheduleSampler):
    def __init__(self, diffusion, iters_all=1e3, mean=300):
        self.name = 'Fast'
        print("Using FastSampler")
        self.diffusion = diffusion
        self.iters_all = iters_all
        self.mean = mean
        gauss = multivariate_normal(mean=mean, cov=1e6)
        x = np.linspace(0, diffusion.num_timesteps, diffusion.num_timesteps)
        self.g = gauss.pdf(x)
        self.g = self.g / np.sum(self.g)

        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

    def sample(self, batch_size, device):
        p = self.weights()
        p = p / np.sum(p)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        # breakpoint()
        # len(p)=4000, p[indices_np]是一个长度为batch_size的数组，每个元素是p中对应的值
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, th.ones_like(weights)

    def update_weights(self, iters):
        rate = min(iters / self.iters_all, 1)
        # breakpoint()
        # if (iters % 500 == 0):
        #     breakpoint()
        # 改进点在这里
        self._weights = np.ones([self.diffusion.num_timesteps]) * (1 - rate + self.g * rate)


class ESSampler(ScheduleSampler):
    def __init__(self, diffusion, iters_all=1e3):
        self.name = 'es'
        print("evolution strategy sampler")
        self.diffusion = diffusion
        self.iters_all = iters_all

        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

    def sample(self, batch_size, device, cand, split_point_num, original_num_steps):
        # 使用插值算法
        # 如果cand是字符串，需要转换成list
        if isinstance(cand, str):
            cand = eval(cand)
        p_at_splitpoints = cand
        interval = original_num_steps // split_point_num
        split_points = [i * interval for i in range(split_point_num + 1)]
        p = np.interp(range(original_num_steps), split_points, p_at_splitpoints)
        normalized_p = p / np.sum(p)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=normalized_p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        # breakpoint()

        return indices, th.ones_like(weights)


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        从本地模型的损失信息中更新采样权重，
        然后执行全局同步以确保所有排名（或进程）具有相同的权重。
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        这是一个抽象方法，需要在子类中实现。
        它用于根据模型的损失信息来更新采样权重。不同的子类可以实现不同的权重更新逻辑。
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


# 基于这个基类实现的具体采样器，用于特定的任务，例如减少目标函数方差并关注损失的第二矩。
# LossSecondMomentResampler 扩展了 LossAwareSampler 并提供了特定的权重更新策略。
class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
