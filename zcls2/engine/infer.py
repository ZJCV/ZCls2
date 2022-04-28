# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:40
@file: infer.py
@author: zj
@description: 
"""

import time

from typing import List
from yacs.config import CfgNode

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config.key_word import KEY_OUTPUT
from ..util.meter import AverageMeter
from ..util.prefetcher import data_prefetcher
from ..util.metric import accuracy
from ..util.distributed import reduce_tensor
from ..util.misc import to_python_float

from zcls2.util import logging

logger = logging.get_logger(__name__)


def validate(cfg: CfgNode, val_loader: DataLoader, model: nn.Module, criterion: nn.Module) -> List:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top_k = cfg.TRAIN.TOP_K
    top_list = [AverageMeter() for _ in top_k]

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(cfg, val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec_list = accuracy(output[KEY_OUTPUT].data, target, topk=top_k)

        if cfg.DISTRIBUTED:
            reduced_loss = reduce_tensor(cfg.NUM_GPUS, loss.data)
            prec_list = [reduce_tensor(cfg.NUM_GPUS, prec) for prec in prec_list]
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        for idx, prec in enumerate(prec_list):
            top_list[idx].update(to_python_float(prec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if cfg.RANK_ID == 0 and i % cfg.PRINT_FREQ == 0:
            logger_str = 'Test: [{0}/{1}] ' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                         'Speed {2:.3f} ({3:.3f}) ' \
                         'Loss {loss.val:.4f} ({loss.avg:.4f}) '.format(
                i, len(val_loader),
                cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.val,
                cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.avg,
                batch_time=batch_time, loss=losses)
            for k, top in zip(top_k, top_list):
                logger_str += f'Prec@{k} {top.val:.3f} ({top.avg:.3f}) '
            logger.info(logger_str)

        input, target = prefetcher.next()

    logger_str = ' * '
    for k, top in zip(top_k, top_list):
        logger_str += f'Prec@{k} {top.avg:.3f} '
    logger.info(logger_str)

    # return top1.avg, top5.avg
    return [top.avg for top in top_list]
