# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:52
@file: dataset.py
@author: zj
@description: Dataset settings
"""

from yacs.config import CfgNode as CN


def add_config(_C: CN) -> None:
    # ---------------------------------------------------------------------------- #
    # DataSet
    # ---------------------------------------------------------------------------- #
    _C.DATASET = CN()
    _C.DATASET.NAME = 'CIFAR100'
    # train data path
    _C.DATASET.TRAIN_ROOT = './data/cifar'
    # test data path
    _C.DATASET.TEST_ROOT = './data/cifar'
