# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:48
@file: lr_scheduler.py
@author: zj
@description: Lr-scheduler settings
"""

from yacs.config import CfgNode as CN


def add_config(_C: CN) -> None:
    # ---------------------------------------------------------------------------- #
    # LR_Scheduler
    # ---------------------------------------------------------------------------- #
    _C.LR_SCHEDULER = CN()
    # LR scheduler type (default: multi_step_lr)
    _C.LR_SCHEDULER.NAME = 'multi_step_lr'
    # Is warmup (default: True)
    _C.LR_SCHEDULER.IS_WARMUP = True
    # Warmup epochs (default: 5)
    _C.LR_SCHEDULER.WARMUP_EPOCH = 5

    # ---------------------------------------------------------------------------- #
    # MultiStepLR
    # ---------------------------------------------------------------------------- #
    _C.LR_SCHEDULER.MULTISTEP_LR = CN()
    _C.LR_SCHEDULER.MULTISTEP_LR.MILESTONES = [30, 60, 80]
    _C.LR_SCHEDULER.MULTISTEP_LR.GAMMA = 0.1
    # If set STEP_SIZE != 0, then omit MILESTONES.
    _C.LR_SCHEDULER.MULTISTEP_LR.STEP_SIZE = 0

    # ---------------------------------------------------------------------------- #
    # CosineAnnealingLR
    # ---------------------------------------------------------------------------- #
    _C.LR_SCHEDULER.COSINE_ANNEALING_LR = CN()
    _C.LR_SCHEDULER.COSINE_ANNEALING_LR.MINIMAL_LR = 1e-6
