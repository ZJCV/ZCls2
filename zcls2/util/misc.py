# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:41
@file: misc.py
@author: zj
@description: 
"""

import os
import torch

from zcls2.util import logging

logger = logging.get_logger(__name__)


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


# Use a local scope to avoid dangling references
def resume(cfg, model, optimizer=None, lr_scheduler=None, device=torch.device('cpu')):
    if os.path.isfile(cfg.RESUME):
        logger.info("=> loading checkpoint '{}'".format(cfg.RESUME))
        # checkpoint = torch.load(cfg.RESUME, map_location=lambda storage, loc: storage.to(device))
        checkpoint = torch.load(cfg.RESUME, map_location=device)
        cfg.TRAIN.START_EPOCH = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        if hasattr(checkpoint, 'best_prec_list'):
            global best_prec_list
            best_prec_list = checkpoint['best_prec_list']
        if hasattr(checkpoint, 'epoch'):
            global best_epoch
            best_epoch = checkpoint['epoch']
        if hasattr(checkpoint, 'optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        if hasattr(checkpoint, 'lr_scheduler'):
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(cfg.RESUME, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(cfg.RESUME))
