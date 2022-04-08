# -*- coding: utf-8 -*-

"""
@date: 2022/4/8 上午10:47
@file: buld.py
@author: zj
@description: 
"""

import torch

from .collate import fast_collate


def build_dataloader(args, cfg, train_dataset, val_dataset, train_sampler, val_sampler, shuffle, memory_format):
    collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

    return train_loader, val_loader
