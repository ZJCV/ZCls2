# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 下午10:24
@file: cosine_annealing_lr.py
@author: zj
@description: CosineAnnealingLR
"""

from typing import Optional

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

__all__ = ["cosine_annealing_lr"]


def cosine_annealing_lr(optimizer: Optimizer,
                        max_epoch: Optional[int] = 90,
                        minimal_lr: Optional[float] = 1e-6) -> CosineAnnealingLR:
    assert isinstance(optimizer, Optimizer)
    # Ensure that the last round is minimal_lr
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch - 1, eta_min=minimal_lr)

    return lr_scheduler
