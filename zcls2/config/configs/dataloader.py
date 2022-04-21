# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:50
@file: dataloader.py
@author: zj
@description: DataLoader settings
"""

from yacs.config import CfgNode as CN


def add_config(_C: CN) -> None:
    # ---------------------------------------------------------------------------- #
    # DataLoader
    # ---------------------------------------------------------------------------- #
    _C.DATALOADER = CN()
    # Batch size per GPU
    _C.DATALOADER.TRAIN_BATCH_SIZE = 256
    _C.DATALOADER.TEST_BATCH_SIZE = 256

    # Refer to [torch Dataloader中的num_workers](https://zhuanlan.zhihu.com/p/69250939)
    _C.DATALOADER.NUM_WORKERS = 4

    # Random sample or sequential sample in train/test stage, default False
    _C.DATALOADER.RANDOM_SAMPLE = True

    # Merges a list of samples to form a mini-batch of Tensor(s).
    # Used when using batched loading from a map-style dataset.
    # Choices: ['fast', 'default']
    _C.DATALOADER.COLLATE_FN = 'fast'
