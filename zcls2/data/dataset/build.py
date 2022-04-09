# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:46
@file: build.py
@author: zj
@description: 
"""

import os

from .general_dataset import GeneralDataset
from .general_dataset_v2 import GeneralDatasetV2
from .mp_dataset import MPDataset

__supported_dataset__ = [
    'GeneralDataset',
    'GeneralDatasetV2',
    'MPDataset'
]


def build_dataset(cfg, train_transform, val_transform):
    dataset_name = cfg.DATASET.NAME
    assert dataset_name in __supported_dataset__, f"{dataset_name} do not support"

    # Data loading code
    traindir = cfg.DATASET.TRAIN_ROOT
    valdir = cfg.DATASET.TEST_ROOT

    if dataset_name == 'GeneralDataset':
        train_dataset = GeneralDataset(
            traindir, transform=train_transform
        )
        val_dataset = GeneralDataset(
            valdir, transform=val_transform
        )
    elif dataset_name == 'GeneralDatasetV2':
        train_dataset = GeneralDatasetV2(
            traindir, transform=train_transform
        )
        val_dataset = GeneralDatasetV2(
            valdir, transform=val_transform
        )
    elif dataset_name == 'MPDataset':
        num_gpus = cfg.NUM_GPUS
        rank_id = cfg.RANK_ID
        epoch = cfg.TRAIN.START_EPOCH

        train_dataset = MPDataset(
            traindir, transform=train_transform, shuffle=True, num_gpus=num_gpus, rank_id=rank_id, epoch=epoch
        )
        val_dataset = MPDataset(
            valdir, transform=val_transform, shuffle=False, num_gpus=num_gpus, rank_id=rank_id, epoch=epoch
        )
    else:
        raise ValueError(f"{dataset_name} do not support")

    return train_dataset, val_dataset
