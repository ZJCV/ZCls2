# -*- coding: utf-8 -*-

"""
@date: 2022/5/3 上午10:08
@file: soft_target_cross_entropy_loss.py
@author: zj
@description: Derived from [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
"""

from typing import Dict

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from zcls2.config.key_word import KEY_OUTPUT

__all__ = ['SoftTargetCrossEntropy', "soft_target_cross_entropy_loss"]


class SoftTargetCrossEntropy(nn.Module):
    """
    Optimized for MixUpd
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: Dict, target: Tensor) -> Tensor:
        x = x[KEY_OUTPUT]
        if len(target.shape) == 1:
            target = F.one_hot(target, num_classes=x.shape[1])
        # print(target.shape, F.log_softmax(x, dim=-1).shape)

        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


soft_target_cross_entropy_loss = SoftTargetCrossEntropy
