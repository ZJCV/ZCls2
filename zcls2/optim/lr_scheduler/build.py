# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:37
@file: build.py
@author: zj
@description: 
"""

from yacs.config import CfgNode
from torch.optim.optimizer import Optimizer

from .multi_step_lr import build_multistep_lr
from .cosine_annealing_lr import build_cosine_annearling_lr

__supported_lr_scheduler__ = [
    'MultiStepLR',
    'CosineAnnealingLR'
]


def adjust_learning_rate(cfg: CfgNode, optimizer: Optimizer, epoch: int, step: int, len_epoch: int) -> None:
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    lr = cfg.OPTIMIZER.LR

    warmup_epoch = cfg.LR_SCHEDULER.WARMUP_EPOCH
    """Warmup"""
    if epoch < warmup_epoch + 1:
        lr = lr * float(1 + step + (epoch - 1) * len_epoch) / (warmup_epoch * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_lr_scheduler(cfg: CfgNode, optimizer: Optimizer):
    lr_scheduler_name = cfg.LR_SCHEDULER.NAME
    assert lr_scheduler_name in __supported_lr_scheduler__

    if lr_scheduler_name == 'MultiStepLR':
        milestones = cfg.LR_SCHEDULER.MULTISTEP_LR.MILESTONES
        gamma = cfg.LR_SCHEDULER.MULTISTEP_LR.GAMMA
        return build_multistep_lr(optimizer, milestones=milestones, gamma=gamma)
    elif lr_scheduler_name == 'CosineAnnealingLR':
        warmup = cfg.LR_SCHEDULER.IS_WARMUP
        warmup_epoch = cfg.LR_SCHEDULER.WARMUP_EPOCH
        max_epoch = cfg.TRAIN.MAX_EPOCH

        return build_cosine_annearling_lr(optimizer,
                                          warmup=warmup, warmup_epoch=warmup_epoch, max_epoch=max_epoch,
                                          minimal_lr=1e-6)
    else:
        raise ValueError(f"{lr_scheduler_name} does not support")
