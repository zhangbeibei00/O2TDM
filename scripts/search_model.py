"""
Train a diffusion model on images.
"""
import argparse
import os
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import random
import torch
import torch.nn as nn
import sys
import time

from tqdm import tqdm
import torchvision.transforms as transforms
import collections

sys.setrecursionlimit(10000)

import functools
import pdb
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler

from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from improved_diffusion.train_util import TrainLoop

import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

print = functools.partial(print, flush=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    Args:
        num_timesteps: The number of diffusion steps in the original process to divide up
        section_counts: Either a list of numbers, or a string containing comma-separated numbers,
                       indicating the step count per section. Special case: use "ddimN" where N
                       is a number of steps to use the striding from the DDIM paper.

    Returns:
        A set of diffusion steps from the original process to use.
    """

    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]  # [10, 20]
    size_per = num_timesteps // len(section_counts)  # 500  1000
    extra = num_timesteps % len(section_counts)  # 0
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)  # 500
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)  # (500 - 1) / (10 - 1) = 55.4
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class FIDStatistics:
    """
    Class for calculating Frechet Inception Distance (FID) between two distributions.
    """
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
                mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
                sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2
        import warnings
        from scipy import linalg
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                    "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                    % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        # 这个类是用来计算两个分布的距离的，这里的距离是两个分布的frechet distance
        # 两个分布的frechet distance是两个分布的均值的欧式距离加上两个分布的协方差矩阵的trace加上两个分布的协方差矩阵的行列式的自然对数

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class EvolutionSearcher(object):
    """
    Evolutionary algorithm for searching optimal timestep schedules.
    """
    def __init__(self, args, model, base_diffusion, time_step):
        """
        Initialize the evolution searcher.
        
        Args:
            args: Configuration arguments
            model: The diffusion model
            base_diffusion: Base diffusion process
            time_step: Number of timesteps to search for
        """
        self.args = args
        self.model = model
        self.base_diffusion = base_diffusion
        
        # Create deep copies to avoid modifying originals during evolution
        import copy
        self.active_diffusion = copy.deepcopy(base_diffusion)
        self.active_model = copy.deepcopy(model)
        self.time_step = time_step
        
        # EA hyperparameters
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        ## tracking variable
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.max_fid = args.max_fid
        self.thres = args.thres
        self.data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
        kwargs = dict(iters_all=args.iters_all, mean=args.mean)
        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, self.active_diffusion, **kwargs)

        self.x0 = args.init_x
        self.x0_1 = args.init_x1

        from evaluations.evaluator_v1 import Evaluator_v1
        import tensorflow.compat.v1 as tf
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        self.evaluator = Evaluator_v1(tf.Session(config=config))
        self.evaluator.warmup()
        import pickle
        f = open(args.ref_path, 'rb')
        self.ref_stats = pickle.load(f)

    def reset_diffusion(self, use_timesteps):
        # 用于重置扩散（diffusion）对象的属性，以便适应特定的时间步（timestep）
        # 转换为集合，以确保它只包含唯一的时间步索引。
        use_timesteps = set(use_timesteps)
        self.active_diffusion.timestep_map = []
        # 将在后续的循环中用于计算新的扩散属性。
        last_alpha_cumprod = 1.0
        new_betas = []

        self.active_diffusion.use_timesteps = set(use_timesteps)
        # 接下来，进入循环，循环遍历基础扩散（base_diffusion）的alphas_cumprod属性，这是一个包含累积α值的数组。
        for i, alpha_cumprod in enumerate(self.base_diffusion.alphas_cumprod):
            if i in use_timesteps:
                # 计算新的β值，这里通过长序列的α值减去前一个时间步的α值来计算β值，用于描述扩散过程。
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)  # 通过长序列的 \overline{alpha} 解 短序列的 \beta
                last_alpha_cumprod = alpha_cumprod
                # 将当前时间步的索引i添加到active_diffusion的timestep_map属性中，以标记激活的时间步索引。
                self.active_diffusion.timestep_map.append(i)

        import numpy as np
        new_betas = np.array(new_betas, dtype=np.float64)

        self.active_diffusion.betas = new_betas
        assert len(new_betas.shape) == 1, "betas must be 1-D"
        assert (new_betas > 0).all() and (new_betas <= 1).all()

        self.active_diffusion.num_timesteps = int(new_betas.shape[0])
        # 更新`active_diffusion`的一些其他属性
        alphas = 1.0 - new_betas  # alpha 递减
        self.active_diffusion.alphas_cumprod = np.cumprod(alphas, axis=0)  # overliane_{x}
        self.active_diffusion.alphas_cumprod_prev = np.append(1.0, self.active_diffusion.alphas_cumprod[
                                                                   :-1])  # alpha[0], alpha[0], alpha[1], ...., alpha[T-1]
        self.active_diffusion.alphas_cumprod_next = np.append(self.active_diffusion.alphas_cumprod[1:],
                                                              0.0)  # alpha[1], alpha[2], ..., alpha[T], alpha[T]
        assert self.active_diffusion.alphas_cumprod_prev.shape == (self.active_diffusion.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.active_diffusion.sqrt_alphas_cumprod = np.sqrt(
            self.active_diffusion.alphas_cumprod)  # \sqrt{\overline{\alpha}}
        self.active_diffusion.sqrt_one_minus_alphas_cumprod = np.sqrt(
            1.0 - self.active_diffusion.alphas_cumprod)  # \sqrt{1 - \overline{\alpha}}
        self.active_diffusion.log_one_minus_alphas_cumprod = np.log(
            1.0 - self.active_diffusion.alphas_cumprod)  # \log{1 - \overline{\alpha}}
        self.active_diffusion.sqrt_recip_alphas_cumprod = np.sqrt(
            1.0 / self.active_diffusion.alphas_cumprod)  # \frac{1}{\sqrt{\overline{\alpha}}}
        self.active_diffusion.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.active_diffusion.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.active_diffusion.posterior_variance = (
                new_betas * (1.0 - self.active_diffusion.alphas_cumprod_prev) / (
                1.0 - self.active_diffusion.alphas_cumprod)  # DDPM 式7 的 \hat{\beta}
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        if len(self.active_diffusion.posterior_variance) > 1:
            self.active_diffusion.posterior_log_variance_clipped = np.log(
                np.append(self.active_diffusion.posterior_variance[1], self.active_diffusion.posterior_variance[1:])
            )
        else:
            self.active_diffusion.posterior_log_variance_clipped = self.active_diffusion.posterior_variance
        self.active_diffusion.posterior_mean_coef1 = (
                new_betas * np.sqrt(self.active_diffusion.alphas_cumprod_prev) / (
                1.0 - self.active_diffusion.alphas_cumprod)
        )
        self.active_diffusion.posterior_mean_coef2 = (
                (1.0 - self.active_diffusion.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.active_diffusion.alphas_cumprod)
        )

    def update_top_k(self, candidates, *, k, key, reverse=False):
        """
        Update the class attribute `keep_top_k` which maintains a list of top k candidates from current search
        
        Args:
            candidates: List of candidates from current iteration
            k: Integer specifying how many top candidates to keep
            key: Function defining how to compare candidates. Applied to each element for sorting.
            reverse: Boolean indicating whether to sort in descending order (default: False)
        """
        assert k in self.keep_top_k
        logger.log('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def sample_active_subnet(self):
        """
        Randomly select timesteps from possible timesteps to generate an active subnet
        """
        original_num_steps = self.base_diffusion.original_num_steps
        use_timestep = [i for i in range(original_num_steps)]
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step]
        return use_timestep

    def sample_point_weight(self):
        """
        Uniformly split timesteps and assign random weights (5-10) to each split point
        """
        split_point_num = self.time_step
        original_num_steps = self.base_diffusion.original_num_steps
        logger.log('split_point_num: {}, original_num_steps: {}'.format(split_point_num, original_num_steps))
        # Define split_point_num evenly spaced points
        interval = original_num_steps // split_point_num
        split_points = [i * interval for i in range(split_point_num + 1)]
        # Generate random weights between 5-10
        weights_at_splitpoints = [random.randint(5, 10) for _ in range(split_point_num + 1)]
        return weights_at_splitpoints

    def is_legal_before_search(self, cand):
        # 用于在执行进化搜索之前检查候选者是否合法。
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        # 已访问
        if 'visited' in info:
            logger.log('cand: {} has visited!'.format(cand))
            return False
        # 未访问就计算fid值，将fid值存入字典
        info['fid'] = self.get_cand_fid(args=self.args, cand=eval(cand))
        logger.log('cand: {}, fid: {}'.format(cand, info['fid']))
        # 写入文件
        # 获取当前日期
        date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        # 保存的文件名为cand_fid_日期.txt
        with open('cand_fid_{}.txt'.format(date), 'a') as file:
            line = 'cand: {}, fid: {}\n'.format(cand, info['fid'])
            file.write(line)

        info['visited'] = True
        return True

    def is_legal(self, cand):
        # 用于检测候选者是否合法。
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logger.log('cand: {} has visited!'.format(cand))
            return False
        info['fid'] = self.get_cand_fid(args=self.args, cand=eval(cand))
        logger.log('cand: {}, fid: {}'.format(cand, info['fid']))
        # 获取当前日期
        date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        # 保存的文件名为cand_fid_日期.txt
        with open('cand_fid_{}.txt'.format(date), 'a') as file:
            line = 'cand: {}, fid: {}\n'.format(cand, info['fid'])
            file.write(line)

        info['visited'] = True
        return True

    def get_cand_fid(self, cand=None, args=None):
        # 使用这个cand的概率分布去训练模型,训练模型5轮，然后采样足够的图片用于计算候选者的fid值。
        t1 = time.time()
        logger.log("creating data loader...")
        train_cand_fid_iter = 100
        logger.log("training {} iters...".format(train_cand_fid_iter))
        # 每次计算新cand序列fid时，先重置模型，再使用cand进行训练然后采样
        import copy
        self.active_diffusion = copy.deepcopy(self.base_diffusion)
        self.active_model = copy.deepcopy(self.model)
        # 设置成训练模式
        self.active_model.train()
        TrainLoop(
            model=self.active_model,
            diffusion=self.active_diffusion,
            data=self.data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=self.schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            # betas=(args.beta1, args.beta2),  # 0.9, 0.999 default
            beta1_decay=args.beta1_decay,
            sample_p=cand,
        ).run_n_loop(train_cand_fid_iter)

        train_time = time.time() - t1
        t1 = time.time()
        # 将模型设置为评估模式，并开始生成图像样本。生成样本的过程可能包括根据策略生成图像，并进行一些后处理操作。
        self.active_model.eval()
        # sample image

        logger.log("sampling...")
        all_images = []
        all_labels = []

        # 将timestep_respaceing设置为250
        args.timestep_respacing = "250"
        # 创建sample_diffusion对象，使用该对象生成样本。
        _, sample_diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        # 生成足够的样本以计算FID值
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            sample_fn = (
                sample_diffusion.p_sample_loop if not args.use_ddim else sample_diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.active_model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log("created " + str(len(all_images) * args.batch_size) + " samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]  # npz文件
        dist.barrier()
        logger.log("sampling complete")
        sample_time = time.time() - t1
        t1 = time.time()

        from evaluations.evaluator_v1 import cal_fid, FIDStatistics
        fid = cal_fid(arr, 64, self.evaluator, ref_stats=self.ref_stats)
        
        if fid < 38:
            # 保存当前的模型节点
            logger.log('fid_less37: {}, cand: {}'.format(fid, cand))
            # 尝试采样1w张图片，计算fid值
            logger.log("sampling...")
            all_images = []
            # 生成1万张样本以计算FID值
            while len(all_images) * args.batch_size < 10000:
                model_kwargs = {}
                sample_fn = (
                    sample_diffusion.p_sample_loop if not args.use_ddim else sample_diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    self.active_model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                logger.log("created " + str(len(all_images) * args.batch_size) + " samples")

            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]  # npz文件
            dist.barrier()
            logger.log("sampling complete")
            sample_time = time.time() - t1
            t1 = time.time()

            from evaluations.evaluator_v1 import cal_fid, FIDStatistics
            fid = cal_fid(arr, 64, self.evaluator, ref_stats=self.ref_stats)
            logger.log('*****************1w_fid: {}, cand: {}**************'.format(fid, cand))
        
        # 将arg.timestep_respacing设置为0
        args.timestep_respacing = "0"
        fid_time = time.time() - t1
        logger.log(
            'train_time: ' + str(train_time) + ', sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))
        return fid

    def get_random_before_search(self, num):
        logger.log('random select ........')
        while len(self.candidates) < num:
            # 随机生成一个采样时间步序列作为candidate
            cand = self.sample_point_weight()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logger.log('random {}/{}'.format(len(self.candidates), num))
        logger.log('random_num = {}'.format(len(self.candidates)))

    def get_random(self, num):
        logger.log('random select ........')
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.log('random {}/{}'.format(len(self.candidates), num))
        logger.log('random_num = {}'.format(len(self.candidates)))

    # 这个方法的目的是执行交叉操作，随机选择两个候选策略进行交叉，生成新的候选策略，并确保生成的候选策略是合法的。
    # 这些新的候选策略可以用于进一步的评估或进化搜索。
    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logger.log('cross ......')
        res = []
        # 设置 `max_iters` 变量为 `cross_num` 的10倍，用于控制最大的交叉尝试次数。
        max_iters = cross_num * 10

        def random_cross():
            # 用于随机执行两个候选策略的交叉操作，生成一个新的候选策略。
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)
            # 遍历 `cand1` 和 `cand2` 中的每个元素，
            # 以一定的概率（这里是50%的概率）从其中一个候选策略中选择对应位置的元素，构建新的候选策略 `new_cand`。
            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])

            return new_cand

        # 循环执行以下步骤，直到 `res` 列表中的候选策略数量达到 `cross_num` 或者 `max_iters` 达到上限：
        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            # 交叉操作，生成一个新的候选策略。
            cand = random_cross()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.log('cross {}/{}'.format(len(res), cross_num))

        logger.log('cross_num = {}'.format(len(res)))
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        # 执行变异操作以生成新的候选策略。
        # - `mutation_num`：一个整数，表示要生成的新候选策略的数量。
        # - `m_prob`：一个浮点数，表示变异操作的概率，即在每个位置上执行变异的概率。
        assert k in self.keep_top_k
        logger.log('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 5

        def random_func():
            # 对当前评分前10的候选策略中的一个进行变异操作，生成一个新的候选策略。
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)
            # 存储可供变异的时间步权重。
            candidates = [3, 4, 5, 6, 7, 8, 9, 10]
            # 遍历 `cand` 中的每个时间步索引，以概率 `m_prob`（即变异概率）决定是否对该时间步执行变异操作。
            # 如果决定执行变异，则随机选择 `candidates` 中的一个时间步索引来替代当前时间步索引。
            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    cand[i] = new_c

            return cand

        # 循环执行以下步骤，直到 `res` 列表中的候选策略数量达到 `mutation_num` 或者 `max_iters` 达到上限：
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.log('mutation {}/{}'.format(len(res), mutation_num))

        logger.log('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x(self, x0, mutation_num, m_prob):
        """
        Mutate initial strategy x0 to generate new candidate strategies
        
        Args:
            x0: Initial strategy string
            mutation_num: Number of new candidates to generate
            m_prob: Mutation probability - probability of mutating each position
        """
        logger.log('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)
            candidates = [3, 4, 5, 6, 7, 8, 9, 10]
            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    cand[i] = new_c
            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logger.log('mutation x0 {}/{}'.format(len(res), mutation_num))

        logger.log('mutation_num = {}'.format(len(res)))
        return res

    def random_search(self):
        """
        Instead of using evolution algorithm, randomly sample strategies before training,
        then train and select the best strategy
        """
        self.get_random_before_search(self.population_num)
        self.update_top_k(
            self.candidates, k=10, key=lambda x: self.vis_dict[x]['fid'])
        for i, cand in enumerate(self.keep_top_k[self.select_num]):
            logger.log('No.{} {} fid = {}'.format(
                i + 1, cand, self.vis_dict[cand]['fid']))

    def load_candidate(self):
        """
        Load candidate strategies from file
        """
        if len(self.candidates) > 0:
            self.candidates = []
        with open('/media/HDD8T/home/fchao/Documents/DC/zhangbeibei/IDDPM/cand_init.txt', 'r') as f:
            for line in f.readlines():
                if line is not None:
                    cand = line.split(":")[1].strip(', fid')
                    score = line.split(":")[2].strip()
                    self.candidates.append(cand)
                    self.vis_dict[cand] = {}
                    self.vis_dict[cand]['fid'] = float(score)
                    self.vis_dict[cand]['visited'] = True
                    logger.log('cand: {}, fid: {}'.format(cand, score))

    def search(self):
        # 是整个进化搜索过程的主要部分。
        logger.log(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        # 根据初始策略 `x0` 是否为空，执行不同的初始化操作：
        if self.x0 != '':
            # 如果 `x0` 不为空，则首先使用 `get_random_before_search` 方法随机选择一半数量的初始候选策略，
            # 并将其存储在 `self.candidates` 中。
            self.get_random_before_search(self.population_num // 3)
            # self.load_candidate()
            # 然后使用 `mutate_init_x` 方法对 `x0` 进行变异操作，生成剩余数量的候选策略，并将其存储在 `self.candidates` 中。
            res = self.mutate_init_x(x0=self.x0, mutation_num=self.population_num // 3, m_prob=0.05)
            self.candidates += res
            res2 = self.mutate_init_x(x0=self.x0_1, mutation_num=self.population_num // 3,
                                      m_prob=0.1)
            self.candidates += res2

        else:
            # 如果以上条件都不满足，则只使用 `get_random_before_search` 方法随机选择初始候选策略，
            # 并将它们存储在 `self.candidates` 中。
            self.get_random_before_search(self.population_num)

        def dec(fid):
            return fid < self.max_fid

        # 循环执行以下操作，直到达到最大迭代次数
        while self.epoch < self.max_epochs:
            logger.log('epoch = {}'.format(self.epoch))

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['fid'])

            logger.log('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.select_num])))
            # 输出当前迭代次数的前 `10` 个最优候选策略的结果，包括策略编号和评分。
            for i, cand in enumerate(self.keep_top_k[self.select_num]):
                logger.log('No.{} {} fid = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['fid']))

            if self.epoch + 1 == self.max_epochs:
                break
            # sys.exit()
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            # self.get_random(self.population_num)  # 变异+杂交凑不足population size的部分重新随机采样
            # rf_features = np.asarray(self.rf_features, dtype='float')
            # self.RandomForestClassifier.fit(rf_features, self.rf_lebal) # refit

            self.epoch += 1


# image_sample_argparser
def create_argparser():
    defaults = dict(
        # sample config
        clip_denoised=True,
        num_samples=500,  
        batch_size=32,
        use_ddim=True,
        model_path="",
        save_dir="",
        time_step=10,  # 分成几个权重点
        # EA search config
        seed=0,
        deterministic=False,
        local_rank=0,
        max_epochs=2,
        select_num=10,
        population_num=10,
        m_prob=0.1,
        crossover_num=5,
        mutation_num=5,
        classifier_path="",
        classifier_scale=1.0,
        max_fid=42.0,
        thres=0.2,
        ref_path='/home/zyx/zbb/improved-diffusion/evaluations/anifaces_ref_stats.pkl',
        ref_spatial_path='/home/zyx/zbb/improved-diffusion/evaluations/anifaces_ref_stats_spatial.pkl',
        init_x='[6, 10, 6, 5, 9, 10, 9, 9, 10, 5, 5]',
        init_x1='[7, 8, 7, 4, 10, 6, 8, 3, 4, 10, 7]',
        use_ddim_init_x=False,
        MASTER_PORT='12345',
        # tarin config
        data_dir='/home/zyx/zbb/improved-diffusion/datasets/anime-faces',
        schedule_sampler="es",  # uniform,es
        lr=2e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        # batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        beta1=0.9,
        beta2=0.999,
        beta1_decay=False,
        iters_all=1e3,
        mean=300,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def save2file(searcher, iters):
    # 关于文件保存策略记录的说明
    # 更新一下保存策略记录的文件，新增一行记录是第几个训练轮次的分隔线
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # 保存的文件名为cand_fid_日期.txt
    with open('cand_fid_{}.txt'.format(date), 'a') as file:
        line = '----------------search_iters:{}---------------------\n'.format(iters)
        file.write(line)

    # 将当前的前10个最优策略保存到文件中
    with open('best_10_cand_{}.txt'.format(date), 'a') as file:
        line = '----------------search_iters:{}---------------------\n'.format(iters)
        file.write(line)
        for j, cand in enumerate(searcher.keep_top_k[searcher.select_num]):
            line = 'No.{} {} fid = {}\n'.format(j + 1, cand, searcher.vis_dict[cand]['fid'])
            file.write(line)

def sample():
    args = create_argparser().parse_args()
    # 设置环境变量

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dist_util.setup_dist()
    # logger.configure(dir='/media/zyx/aniface/ckpt/log')
    # 输出参数
    logger.log(str(args))
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.model_path != "":
        logger.log(f"loading pretrained model {args.model_path}...")
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location='cpu')
        )

    model.to(dist_util.dev())
    kwargs = dict(iters_all=args.iters_all, mean=args.mean)


def main():
    args = create_argparser().parse_args()
    # 设置环境变量

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dist_util.setup_dist()
    # logger.configure(dir='/media/zyx/aniface/ckpt/log')
    # 输出参数
    logger.log(str(args))
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.model_path != "":
        logger.log(f"loading pretrained model {args.model_path}...")
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location='cpu')
        )

    model.to(dist_util.dev())
    kwargs = dict(iters_all=args.iters_all, mean=args.mean)

    total_search_iters = 1
    for i in range(total_search_iters):
        # 直接加载预训练模型在这一步开始搜索时间步概率分布
        t = time.time()
        searcher = EvolutionSearcher(args, model=model, base_diffusion=diffusion, time_step=args.time_step)
        # searcher.search()
        # # 阶段性保存该轮训练的cand记录和最优cand记录
        # save2file(searcher, i)
        # best_candidate = searcher.keep_top_k[searcher.select_num][0]
        # # # 更新搜索器的初始策略
        # searcher.x0 = best_candidate
        # searcher.x0_1 = searcher.keep_top_k[searcher.select_num][1]

        best_candidate = '[7, 8, 7, 4, 10, 6, 5, 3, 3, 4, 7]'  # cifar10
        logger.log('best candidate: {}'.format(best_candidate))

        # 使用最优策略训练模型10000轮
        train_iters = 10000
        logger.log('training {} iters...'.format(train_iters))
        # breakpoint()
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=searcher.data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=searcher.schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            sample_p=eval(best_candidate),
        ).run_n_loop(train_iters)
        # 切换为eval模式
        model.eval()
        # 设置timestep_respaceing为250
        args.timestep_respacing = "250"
        # 创建sample_diffusion对象，使用该对象生成样本。
        _, sample_diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        # 生成足够的样本以计算FID值
        all_images = []

        while len(all_images) * args.batch_size < 32:
            sample_fn = (
                sample_diffusion.p_sample_loop if not args.use_ddim else sample_diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log("created " + str(len(all_images) * args.batch_size) + " samples")

        # 计算FID值
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]  # npz文件

        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")
        # 将arg.timestep_respacing设置为0
        args.timestep_respacing = "0"

        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            # 文件命名为当前时间
            date = time.strftime('%m-%d-%H', time.localtime())
            out_path = os.path.join(logger.get_dir(), f"samples_{date}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)

        from evaluations.evaluator_v1 import cal_evalutions, FIDStatistics
        fid, sfid, iScore = cal_evalutions(arr, 64, searcher.evaluator, args.ref_path, args.ref_spatial_path)
        logger.log('fid: {}'.format(fid))
        logger.log('sfid: {}'.format(sfid))
        logger.log('inception Score: {}'.format(iScore))
        logger.log('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))


def cal_model_fid(model_path, sample_num):
    args = create_argparser().parse_args()
    args.model_path = model_path
    # 设置环境变量

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dist_util.setup_dist()
    # logger.configure(dir='/media/HDD8T/tmp')

    logger.log(str(args))
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log(f"loading pretrained model {args.model_path}...")

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location='cpu')
    )

    model.to(dist_util.dev())

    kwargs = dict(iters_all=args.iters_all, mean=args.mean)

    model.eval()
    # 设置timestep_respaceing为250
    args.timestep_respacing = "250"
    # 创建sample_diffusion对象，使用该对象生成样本。
    _, sample_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # 生成足够的样本以计算FID值
    all_images = []

    while len(all_images) * args.batch_size < sample_num:
        sample_fn = (
            sample_diffusion.p_sample_loop if not args.use_ddim else sample_diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log("created " + str(len(all_images) * args.batch_size) + " samples")
    # 计算FID值
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]  # npz文件
    dist.barrier()
    logger.log("sampling complete")
    from evaluations.evaluator_v1 import cal_evalutions, FIDStatistics
    searcher = EvolutionSearcher(args, model=model, base_diffusion=diffusion, time_step=args.time_step)
    fid, sfid, iScore = cal_evalutions(arr, 64, searcher.evaluator, args.ref_path, args.ref_spatial_path)
    logger.log('fid: {}'.format(fid))
    logger.log('sfid: {}'.format(sfid))
    logger.log('inception Score: {}'.format(iScore))
    return fid

if __name__ == "__main__":
    main()
    # path = '/media/zyx/tiny64/ckpt/original/search_result/10-30w'
    # for root, dirs, files in os.walk(path):
    #     for dir in dirs:
    #         model_path = os.path.join(root, dir, 'model000000_result.pt')
    #         print(model_path)
    #         print("1k_fid")
    #         model_fid_1k=cal_model_fid(model_path,1000)
    #         print("1w_fid")
    #         model_fid_1w=cal_model_fid(model_path,10000)
    # cal_model_fid("/media/zyx/tiny64/ckpt/search_result/11-29-13-03/model100000_result.pt",1000)


