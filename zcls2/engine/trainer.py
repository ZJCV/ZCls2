# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:34
@file: trainer.py
@author: zj
@description: 
"""

import time

from typing import Optional
from yacs.config import CfgNode

from timm.data import Mixup

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..config.key_word import KEY_OUTPUT
from ..util.meter import AverageMeter
from ..util.prefetcher import data_prefetcher
from ..util.metric import accuracy
from ..util.distributed import reduce_tensor
from ..util.misc import to_python_float
from ..optim.lr_scheduler.build import adjust_learning_rate

from zcls2.util import logging

logger = logging.get_logger(__name__)


def train(cfg: CfgNode, train_loader: DataLoader,
          model: nn.Module,
          criterion: nn.Module,
          optimizer: Optimizer,
          epoch: Optional[int] = 1,
          mixup_fn: Optional[Mixup] = None) -> None:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top_k = cfg.TRAIN.TOP_K
    top_list = [AverageMeter() for _ in top_k]

    # switch to train mode
    model.train()
    end = time.time()

    # See https://pytorch.org/docs/stable/amp.html#gradient-scaling
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    warmup = cfg.LR_SCHEDULER.IS_WARMUP
    warmup_epoch = cfg.LR_SCHEDULER.WARMUP_EPOCH

    prefetcher = data_prefetcher(cfg, train_loader)
    samples, targets = prefetcher.next()
    i = 0
    while samples is not None:
        i += 1

        if warmup and epoch < warmup_epoch + 1:
            adjust_learning_rate(cfg, optimizer, epoch, i, len(train_loader))

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # See https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
        # Runs the forward pass with autocasting.
        with autocast():
            # compute output
            output = model(samples)
            loss = criterion(output, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        if i % cfg.PRINT_FREQ == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            if mixup_fn is None and cfg.TRAIN.CALCULATE_ACCURACY:
                prec_list = accuracy(output[KEY_OUTPUT].data, targets, topk=top_k)
            else:
                prec_list = None

            # Average loss and accuracy across processes for logging
            if cfg.DISTRIBUTED:
                reduced_loss = reduce_tensor(cfg.NUM_GPUS, loss.data)
                if prec_list is not None:
                    prec_list = [reduce_tensor(cfg.NUM_GPUS, prec) for prec in prec_list]
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), samples.size(0))
            if prec_list is not None:
                for idx, prec in enumerate(prec_list):
                    top_list[idx].update(to_python_float(prec), samples.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / cfg.PRINT_FREQ)
            end = time.time()

            if cfg.RANK_ID == 0:
                logger_str = 'Epoch: [{0}/{1}][{2}/{3}] ' \
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                             'Speed {4:.3f} ({5:.3f}) ' \
                             'Lr {lr:.10f} ' \
                             'Loss {loss.val:.10f} ({loss.avg:.4f}) '.format(
                    epoch, cfg.TRAIN.MAX_EPOCH, i, len(train_loader),
                    cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.val,
                    cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.avg,
                    batch_time=batch_time,
                    lr=optimizer.param_groups[0]['lr'],
                    loss=losses)
                for k, top in zip(top_k, top_list):
                    logger_str += f'Prec@{k} {top.val:.3f} ({top.avg:.3f}) '
                logger.info(logger_str)
        samples, targets = prefetcher.next()
