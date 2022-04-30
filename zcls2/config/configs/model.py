# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:49
@file: model.py
@author: zj
@description: Model settings
"""

from yacs.config import CfgNode as CN


def add_config(_C: CN) -> None:
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    _C.MODEL = CN()
    # model architecture (default: resnet18)
    _C.MODEL.ARCH = 'resnet18'
    # use pre-trained model
    _C.MODEL.PRETRAINED = True
    # number of model output (default: 100 for CIFAR100)
    _C.MODEL.NUM_CLASSES = 100
    # enabling apex sync BN.
    _C.MODEL.SYNC_BN = False

    # ---------------------------------------------------------------------------- #
    # criterion
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CRITERION = CN()
    _C.MODEL.CRITERION.NAME = 'CrossEntropyLoss'
    # mean or sum
    _C.MODEL.CRITERION.REDUCTION = 'mean'
    # Label smoothing (default: 0.)
    _C.MODEL.CRITERION.LABEL_SMOOTHING = 0.
