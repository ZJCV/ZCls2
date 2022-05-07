# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午5:07
@file: cross_entropy_loss.py
@author: zj
@description: 
"""
from typing import Optional, Dict

from torch import nn, Tensor
from zcls2.config.key_word import KEY_OUTPUT

__all__ = ['CrossEntropyLoss', 'cross_entropy_loss']


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None,
                 reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, input_dict: Dict, target: Tensor) -> Tensor:
        inputs = input_dict[KEY_OUTPUT]

        return super().forward(inputs, target)


cross_entropy_loss = CrossEntropyLoss
