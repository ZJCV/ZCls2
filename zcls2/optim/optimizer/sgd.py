# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午5:00
@file: sgd.py
@author: zj
@description: 
"""
from typing import Optional

from torch.optim.optimizer import Optimizer
from torch.optim import SGD

__all__ = ["sgd"]


def sgd(groups: list,
        lr: Optional[float] = 1e-3,
        momentum: Optional[float] = 0.9,
        weight_decay: Optional[float] = 1e-4) -> Optimizer:
    return SGD(groups,
               lr=lr,
               momentum=momentum,
               weight_decay=weight_decay)
