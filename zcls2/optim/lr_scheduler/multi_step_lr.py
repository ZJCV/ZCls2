# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 下午10:42
@file: multi_step_lr.py
@author: zj
@description: MultiStepLR
"""

from typing import Optional

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

__all__ = ["multi_step_lr"]


def multi_step_lr(optimizer: Optimizer,
                  milestones: Optional[list] = None,
                  gamma: Optional[float] = 0.1,
                  ) -> MultiStepLR:
    assert isinstance(optimizer, Optimizer)
    if milestones is None:
        milestones = [30, 60, 80]

    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
