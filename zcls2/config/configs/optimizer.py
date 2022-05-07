# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:49
@file: optimizer.py
@author: zj
@description: Optimizer settings
"""

from yacs.config import CfgNode as CN


def add_config(_C: CN) -> None:
    # ---------------------------------------------------------------------------- #
    # Optimizer
    # ---------------------------------------------------------------------------- #
    _C.OPTIMIZER = CN()
    # Optimizer type (default: sgd)
    _C.OPTIMIZER.NAME = 'sgd'
    # Initial learning rate.
    # Will be scaled by <global batch size>/256: lr = lr * float(train_batch_size * args.world_size) / 256.
    _C.OPTIMIZER.LR = 1e-1
    _C.OPTIMIZER.MOMENTUM = 0.9
    # ---------------------------------------------------------------------------- #
    # Weight Decay
    # ---------------------------------------------------------------------------- #
    _C.OPTIMIZER.WEIGHT_DECAY = CN()
    # weight decay (default: 1e-4)
    _C.OPTIMIZER.WEIGHT_DECAY.DECAY = 1e-4
    _C.OPTIMIZER.WEIGHT_DECAY.NO_BIAS = True
    _C.OPTIMIZER.WEIGHT_DECAY.NO_NORM = True
