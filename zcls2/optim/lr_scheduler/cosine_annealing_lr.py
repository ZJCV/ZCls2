# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 下午10:24
@file: cosine_annealing_lr.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer


def build_cosine_annearling_lr(optimizer, warmup=True, warmup_epoch=5, max_epoch=90, minimal_lr=1e-6):
    assert isinstance(optimizer, Optimizer)

    if warmup:
        max_epoch = max_epoch - warmup_epoch
    # Ensure that the last round is minimal_lr
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - 1, eta_min=minimal_lr)

    return lr_scheduler
