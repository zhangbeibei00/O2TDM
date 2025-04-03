import copy
import functools
import os
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler, ESSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def beta1_scheduler(iters, beta1_init, iters_all=1e6, minimum=0.4):
    beta1 = beta1_init * (1 - iters / iters_all) / (1 - beta1_init * iters / iters_all)
    return max(beta1, minimum)


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            betas=(0.9, 0.999),
            beta1_decay=False,
            sample_p='',
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        # 动量衰减
        self.beta1_decay = beta1_decay
        self.betas = betas

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self.sample_p = sample_p
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        # 这段代码片段处理了训练中的优化器状态和指数移动平均参数的加载。如果从之前的训练中恢复，它会加载相应的状态；如果从头开始训练，它会初始化相应的参数。
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
                param_group['betas'] = betas
            # self.opt.param_groups[0]['betas'] = betas
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
        # 这段代码片段涉及到分布式数据并行（Distributed Data Parallel，DDP）的设置。DDP 是一种用于在多个 GPU 上进行模型训练的技术，
        # 其中模型的参数被分发到多个设备上，并行地计算和更新梯度。这有助于提高训练速度和扩展性。
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0: 额外注释掉的，不知道什么意思
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        # dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        # 这个条件检查是否达到了学习率退火的步数或者是否达到了预设的步数。
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            if self.beta1_decay and self.step % self.log_interval == 0:
                beta1, beta2 = self.opt.param_groups[0]['betas']
                _beta1 = beta1_scheduler(self.step + self.resume_step, self.betas[0])
                lr_scale = (1 - _beta1) / (1 - self.betas[0])
                for param_group in self.opt.param_groups:
                    param_group["betas"] = (_beta1, beta2)
                    param_group["lr"] = self.lr / lr_scale

            # 从数据生成器 self.data 中获取一个批次数据 batch 和相关的条件数据 cond。
            batch, cond = next(self.data)
            # 运行一个训练步骤，传递批次数据和条件数据。这个步骤涉及到模型的前向传播、反向传播以及参数更新。
            self.run_step(batch, cond)
            # 如果 self.step 是 self.log_interval 的倍数，调用 logger.dumpkvs()。这可能是用于记录或打印一些训练中的指标或日志信息。

            # 如果 self.step 是 self.save_interval 的倍数，调用 self.save()。这可能是用于保存模型的检查点。
            if self.step % self.log_interval == 0:
                if self.schedule_sampler.name != 'Uniform':
                    logger.log(self.schedule_sampler.weights())
                logger.dumpkvs()
                logger.log(time.asctime(time.localtime(time.time())))
                # logger.log(f'image size: {self.model.image_size}')
                _betas = self.opt.param_groups[0]['betas']
                logger.log(f'betas: {_betas}')
                logger.log(f'sampler: {self.schedule_sampler.name}')

                if self.schedule_sampler.name == 'Fast':
                    logger.log(f'end: {self.schedule_sampler.iters_all}, mean: {self.schedule_sampler.mean}')
                    iters = self.step + self.resume_step
                    self.schedule_sampler.update_weights(iters)

            if self.step % self.save_interval == 0:
                # self.save()
                self.save_result()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_n_loop(self, n):
        # 这个条件检查是否达到了学习率退火的步数或者是否达到了预设的步数。
        iters_left = n
        while (
                iters_left > 0
                and (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps)
        ):
            iters_left = iters_left - 1
            if iters_left % 1000 == 0:
                logger.log(f'iters_left: {iters_left}')
                logger.log(f'using sample_p: {self.sample_p}')
                logger.dumpkvs()
                logger.log(time.asctime(time.localtime(time.time())))
                logger.log(f'sampler: {self.schedule_sampler.name}')
            # 从数据生成器 self.data 中获取一个批次数据 batch 和相关的条件数据 cond。
            batch, cond = next(self.data)
            # 运行一个训练步骤，传递批次数据和条件数据。这个步骤涉及到模型的前向传播、反向传播以及参数更新。
            self.run_step(batch, cond)
            # 当训练步数达到一定值时，保存模型的检查点。
            if  n >= 1000 and iters_left % self.save_interval == 0:
                self.save_result()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        # 这个条件语句检查是否启用了混合精度训练（use_fp16 为真）。
        # 如果启用了混合精度训练，调用 optimize_fp16 方法，该方法可能会使用 16 位浮点数格式（half precision）来进行参数更新。
        # 如果未启用混合精度训练，则调用 optimize_normal 方法，该方法可能会使用标准的 32 位浮点数格式来进行参数更新。
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # 对微批次进行时间步骤采样，得到时间步骤 t 和 t对应的权重 weights。
            # t.shape=64,这里的weights是每个t对应的weight,全是1,重要性采样
            if (self.schedule_sampler.name == 'es'):
                t, weights = self.schedule_sampler.sample(batch_size=micro.shape[0], device=dist_util.dev(),
                                                          cand=self.sample_p, split_point_num=10,
                                                          original_num_steps=self.diffusion.num_timesteps)
            else:
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            # 根据是否为最后一个微批次或者是否使用了分布式数据并行进行不同的处理：
            #
            # 如果是最后一个微批次或者没有使用分布式数据并行 (last_batch or not self.use_ddp)，直接计算损失。
            # 否则，使用 with self.ddp_model.no_sync(): 包装，确保在分布式环境中只有部分模型参数同步。
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # 这个条件检查 self.schedule_sampler 是否是 LossAwareSampler 类的实例。
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            # 这里的weight,是每个t对应的weight,重要性采样
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        # self._log_grad_norm(): 调用 _log_grad_norm 方法，该方法可能用于记录或打印当前梯度的范数（或其他相关信息）。
        # 梯度范数是一个指示梯度大小的量，对于监控训练的稳定性和梯度爆炸等问题很有用。
        #
        # self._anneal_lr(): 调用 _anneal_lr 方法，该方法可能用于动态调整学习率。
        # 学习率退火是一种常见的优化策略，可以帮助模型更好地收敛。
        #
        # self.opt.step(): 执行模型参数的一步梯度下降，即更新模型参数。
        # 这里假设 self.opt 是一个 PyTorch 优化器对象，它包含了用于更新模型参数的梯度下降算法（例如随机梯度下降）。
        #
        # 遍历 self.ema_rate 和 self.ema_params，对每个参数进行指数移动平均（EMA）的更新。
        # update_ema 可能是一个用于执行指数移动平均的函数。
        # 指数移动平均是一种平滑序列的方法，常用于模型参数的平均更新，尤其是在训练过程中。
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        # 这段代码定义了一个用于计算和记录模型参数梯度范数的辅助函数
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)
        _lr = self.opt.param_groups[0]['lr']
        logger.logkv('lr', _lr)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    # 保存使用最优的candidate训练10000轮的结果
    def save_result(self):
        def save_checkpoint(rate, params, save_path):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}_result.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}_result.pt"

                with bf.BlobFile(bf.join(save_path, filename), "wb") as f:
                    th.save(state_dict, f)

        # 获取当前路径
        path = os.getcwd()
        # 保存结果的文件夹为当前路径下的search_result文件夹
        # 在search_result文件夹下设置当前日期的文件夹，日期为月-日-小时-分钟

        save_path = os.path.join('/data/zyx/zbb/O2TDM', 'search_result', time.strftime("%m-%d-%H-%M", time.localtime()))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_checkpoint(0, self.master_params, save_path)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params, save_path)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(save_path, f"opt{(self.step + self.resume_step):06d}_result.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
