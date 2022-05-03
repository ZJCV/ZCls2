# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:54
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn
from yacs.config import CfgNode

from . import cross_entropy_loss, large_margin_softmax_loss, soft_target_cross_entropy_loss

__all__ = cross_entropy_loss.__all__ + large_margin_softmax_loss.__all__ + soft_target_cross_entropy_loss.__all__


def build_criterion(cfg: CfgNode) -> nn.Module:
    loss_name = cfg.MODEL.CRITERION.NAME
    reduction = cfg.MODEL.CRITERION.REDUCTION

    assert loss_name in __all__

    if loss_name in cross_entropy_loss.__all__:
        label_smoothing = cfg.MODEL.CRITERION.LABEL_SMOOTHING
        return cross_entropy_loss.__dict__[loss_name](reduction=reduction, label_smoothing=label_smoothing)
    elif loss_name in large_margin_softmax_loss.__all__:
        return large_margin_softmax_loss.__dict__[loss_name](reduction=reduction)
    elif loss_name in soft_target_cross_entropy_loss.__all__:
        return soft_target_cross_entropy_loss.__dict__[loss_name]()
    else:
        raise ValueError(f"{loss_name} does not support")
