# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:48
@file: lr_scheduler.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # LR_Scheduler
    # ---------------------------------------------------------------------------- #
    _C.LR_SCHEDULER = CN()
    # LR scheduler type (default: MultiStepLR)
    _C.LR_SCHEDULER.NAME = 'MultiStepLR'
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

    # ---------------------------------------------------------------------------- #
    # CosineAnnealingLR
    # ---------------------------------------------------------------------------- #
    _C.LR_SCHEDULER.COSINE_ANNEALING_LR = CN()
    _C.LR_SCHEDULER.COSINE_ANNEALING_LR.MINIMAL_LR = 1e-6