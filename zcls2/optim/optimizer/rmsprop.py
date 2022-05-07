# -*- coding: utf-8 -*-

"""
@date: 2022/4/30 下午1:22
@file: rmsprop.py
@author: zj
@description: 
"""
from typing import Optional

from torch.optim.optimizer import Optimizer
from torch.optim import RMSprop

__all__ = ["rmsprop"]


def rmsprop(groups: list,
            lr: Optional[float] = 1e-2,
            momentum: Optional[float] = 0.,
            weight_decay: Optional[float] = 0.) -> Optimizer:
    return RMSprop(groups,
                   lr=lr,
                   momentum=momentum,
                   weight_decay=weight_decay)
