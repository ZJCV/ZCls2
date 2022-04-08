# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:57
@file: build.py
@author: zj
@description: 
"""

from . import resnet, ghostnet

from zcls2.util import logging

logger = logging.get_logger(__name__)

__supported_model__ = resnet.__supported_model__ + ghostnet.__supported_model__


def build_model(cfg, memory_format):
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
    elif model_arch in resnet.__supported_model__:
        model = resnet.get_resnet(pretrained=is_pretrained, num_classes=num_classes, arch=model_arch)
    else:
        raise ValueError(f"{model_arch} does not support")

    if sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    return model
