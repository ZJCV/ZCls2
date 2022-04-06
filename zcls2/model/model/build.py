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


def build_model(args, memory_format):
    assert args.arch in __supported_model__, f"{args.arch} do not in model list"

    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
    else:
        logger.info("=> creating model '{}'".format(args.arch))

    if args.arch in ghostnet.__supported_model__:
        model = ghostnet.get_ghostnet(pretrained=args.pretrained, num_classes=args.num_classes, arch=args.arch)
    elif args.arch in resnet.__supported_model__:
        model = resnet.get_resnet(pretrained=args.pretrained, num_classes=args.num_classes, arch=args.arch)
    else:
        raise ValueError(f"{args.arch} does not support")

    if args.sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    return model
