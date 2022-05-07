# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:37
@file: build.py
@author: zj
@description: adjust_learning_rate and build_lr_scheduler
"""

from yacs.config import CfgNode
from torch.optim.optimizer import Optimizer

from . import multi_step_lr, cosine_annealing_lr

__all__ = ["adjust_learning_rate", "build_lr_scheduler"]


def adjust_learning_rate(cfg: CfgNode, optimizer: Optimizer, epoch: int, step: int, len_epoch: int) -> None:
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    lr = cfg.OPTIMIZER.LR

    warmup_epoch = cfg.LR_SCHEDULER.WARMUP_EPOCH
    # Warmup
    if epoch < warmup_epoch + 1:
        lr = lr * float(1 + step + (epoch - 1) * len_epoch) / (warmup_epoch * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_lr_scheduler(cfg: CfgNode, optimizer: Optimizer):
    lr_scheduler_name = cfg.LR_SCHEDULER.NAME

    warmup = cfg.LR_SCHEDULER.IS_WARMUP
    warmup_epoch = cfg.LR_SCHEDULER.WARMUP_EPOCH
    max_epoch = cfg.TRAIN.MAX_EPOCH

    if warmup:
        max_epoch = max_epoch - warmup_epoch

    if lr_scheduler_name in multi_step_lr.__all__:
        milestones = cfg.LR_SCHEDULER.MULTISTEP_LR.MILESTONES
        gamma = cfg.LR_SCHEDULER.MULTISTEP_LR.GAMMA
        step_size = cfg.LR_SCHEDULER.MULTISTEP_LR.STEP_SIZE
        if step_size != 0:
            milestones = list(range(0, max_epoch, step_size))

        lr_scheduler = multi_step_lr.__dict__[lr_scheduler_name](optimizer, milestones=milestones, gamma=gamma)
    elif lr_scheduler_name in cosine_annealing_lr.__all__:
        lr_scheduler = cosine_annealing_lr.__dict__[lr_scheduler_name](optimizer,
                                                                       max_epoch=max_epoch,
                                                                       minimal_lr=1e-6)
    else:
        raise ValueError(f"{lr_scheduler_name} does not support")

    return lr_scheduler
