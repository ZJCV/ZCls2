# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:47
@file: build.py
@author: zj
@description: 
"""
from typing import Tuple, Any

from yacs.config import CfgNode

from torch.utils.data.distributed import DistributedSampler


def build_sampler(cfg: CfgNode, train_dataset, val_dataset) -> Tuple[Any, Any]:
    train_sampler = None
    val_sampler = None
    if cfg.DISTRIBUTED:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    return train_sampler, val_sampler
