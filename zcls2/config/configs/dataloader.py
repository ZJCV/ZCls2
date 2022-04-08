# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:50
@file: dataloader.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # DataLoader
    # ---------------------------------------------------------------------------- #
    _C.DATALOADER = CN()
    # batch size per GPU
    _C.DATALOADER.TRAIN_BATCH_SIZE = 256
    _C.DATALOADER.TEST_BATCH_SIZE = 256

    # refert to [torch Dataloader中的num_workers](https://zhuanlan.zhihu.com/p/69250939)
    _C.DATALOADER.NUM_WORKERS = 4

    # random sample or sequential sample in train/test stage, default False
    _C.DATALOADER.RANDOM_SAMPLE = True
