# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:54
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn
from yacs.config import CfgNode

from .cross_entropy_loss import build_cross_entropy_loss
from .large_margin_softmax_loss import build_large_margin_softmax_loss

__supported_criterion__ = [
    'CrossEntropyLoss',
    'LargeMarginSoftmaxV1'
]


def build_criterion(cfg: CfgNode) -> nn.Module:
    loss_name = cfg.MODEL.CRITERION.NAME
    reduction = cfg.MODEL.CRITERION.REDUCTION

    assert loss_name in __supported_criterion__

    if loss_name == 'CrossEntropyLoss':
        return build_cross_entropy_loss(reduction=reduction)
    elif loss_name == 'LargeMarginSoftmaxV1':
        return build_large_margin_softmax_loss(reduction=reduction)
    else:
        raise ValueError(f"{loss_name} does not support")
