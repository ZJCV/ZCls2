# -*- coding: utf-8 -*-

"""
@date: 2022/4/8 上午10:47
@file: buld.py
@author: zj
@description: 
"""

import torch

from .collate import fast_collate


def build_dataloader(cfg, train_dataset, val_dataset, train_sampler, val_sampler, shuffle, memory_format):
    train_batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE
    test_batch_size = cfg.DATALOADER.TEST_BATCH_SIZE
    num_workers = cfg.DATALOADER.NUM_WORKERS

    collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

    return train_loader, val_loader
