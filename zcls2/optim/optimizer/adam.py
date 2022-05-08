# -*- coding: utf-8 -*-

"""
@date: 2022/5/8 下午1:53
@file: adam.py
@author: zj
@description: 
"""
from typing import Optional

from torch.optim.optimizer import Optimizer
from torch.optim import Adam

__all__ = ["adam"]


def adam(groups: list,
         lr: Optional[float] = 1e-3,
         weight_decay: Optional[float] = 0.) -> Optimizer:
    return Adam(groups,
                lr=lr,
                weight_decay=weight_decay)
