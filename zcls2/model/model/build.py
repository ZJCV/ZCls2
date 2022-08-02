# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:57
@file: build.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from yacs.config import CfgNode

from zcls2.util import logging
from . import resnet, ghostnet, mobilenet, efficientnet, efficientnet_lite

logger = logging.get_logger(__name__)

__all__ = ["build_model"]


def build_model(cfg: CfgNode, device: torch.device = torch.device('cpu')) -> nn.Module:
    model_arch = cfg.MODEL.ARCH
    is_pretrained = cfg.MODEL.PRETRAINED
    num_classes = cfg.MODEL.NUM_CLASSES
    sync_bn = cfg.MODEL.SYNC_BN

    # create model
    if is_pretrained:
        logger.info("=> using pre-trained model '{}'".format(model_arch))
    else:
        logger.info("=> creating model '{}'".format(model_arch))

    if model_arch in ghostnet.__all__:
        model = ghostnet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes)
    elif model_arch in resnet.__all__:
        model = resnet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes)
    elif model_arch in mobilenet.__all__:
        model = mobilenet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes)
    elif model_arch in efficientnet.__all__:
        model = efficientnet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes)
    elif model_arch in efficientnet_lite.__all__:
        model = efficientnet_lite.__dict__[model_arch](num_classes=num_classes)
    else:
        raise ValueError(f"{model_arch} does not support")

    if sync_bn:
        logger.info("using synced BN")
        # See https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # See https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
    if cfg.CHANNELS_LAST:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    # Same as Apex setting
    model = model.to(device, memory_format=memory_format)

    if cfg.DISTRIBUTED:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)

    return model
