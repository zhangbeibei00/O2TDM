"""
Train a diffusion model on images.
"""

import argparse
import pdb
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import os

# 设置环境变量
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['RDMAV_FORK_SAFE'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.model_dir != "None":
        logger.log(f"loading pretrained model {args.model_dir}")
        model.load_state_dict(dist_util.load_state_dict(args.model_dir, map_location='cpu'))

    model.to(dist_util.dev())
    kwargs = dict(iters_all=args.iters_all, mean=args.mean)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, **kwargs)
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        betas=(args.beta1, args.beta2),  # 0.9, 0.999 default
        beta1_decay=args.beta1_decay,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        model_dir= "None",
        schedule_sampler="fast",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=100000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        local_rank=0,
        beta1=0.9,
        beta2=0.999,
        beta1_decay=False,
        iters_all=200000,
        mean=300,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
