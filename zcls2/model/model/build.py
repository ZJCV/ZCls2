# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:57
@file: build.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from yacs.config import CfgNode

from . import resnet, ghostnet, mobilenet

from zcls2.util import logging

logger = logging.get_logger(__name__)

__supported_model__ = resnet.__all__ + ghostnet.__supported_model__ + mobilenet.__all__


def build_model(cfg: CfgNode, device: torch.device = torch.device('cpu')) -> nn.Module:
    model_arch = cfg.MODEL.ARCH
    is_pretrained = cfg.MODEL.PRETRAINED
    num_classes = cfg.MODEL.NUM_CLASSES
    sync_bn = cfg.MODEL.SYNC_BN

    assert model_arch in __supported_model__

    # create model
    if is_pretrained:
        logger.info("=> using pre-trained model '{}'".format(model_arch))
    else:
        logger.info("=> creating model '{}'".format(model_arch))

    if model_arch in ghostnet.__supported_model__:
        model = ghostnet.get_ghostnet(pretrained=is_pretrained, num_classes=num_classes, arch=model_arch)
    elif model_arch in resnet.__all__:
        model = resnet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes)
    elif model_arch in mobilenet.__all__:
        model = mobilenet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"{model_arch} does not support")

    if sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.to(device)

    return model
