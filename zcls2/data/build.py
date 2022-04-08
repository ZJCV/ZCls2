# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:30
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data import IterableDataset

from .transform.build import build_transform
from .dataset.build import build_dataset
from .sampler.build import build_sampler
from .dataloader.build import build_dataloader


def build_data(args, cfg, memory_format):
    train_transform, val_transform = build_transform(cfg)
    train_dataset, val_dataset = build_dataset(args, cfg, train_transform, val_transform)

    if isinstance(train_dataset, IterableDataset):
        train_sampler, val_sampler = None, None
        shuffle = False
    else:
        train_sampler, val_sampler = build_sampler(args, train_dataset, val_dataset)
        shuffle = train_sampler is None

    train_loader, val_loader = build_dataloader(cfg,
                                                train_dataset, val_dataset, train_sampler, val_sampler,
                                                shuffle, memory_format)

    return train_sampler, train_loader, val_loader
