# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:49
@file: model.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    _C.MODEL = CN()
    # model architecture (default: resnet18)
    _C.MODEL.ARCH = 'resnet18'
    # use pre-trained model
    _C.MODEL.PRETRAINED = False
    # number of model output (default: 1000)
    _C.MODEL.NUM_CLASSES = 1000
    # enabling apex sync BN.
    _C.MODEL.SYNC_BN = False
