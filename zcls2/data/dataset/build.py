# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:46
@file: build.py
@author: zj
@description: 
"""

from .general_dataset import GeneralDataset
from .general_dataset_v2 import GeneralDatasetV2
from .mp_dataset import MPDataset

__supported_dataset__ = [
    'GeneralDataset',
    'GeneralDatasetV2',
    'MPDataset'
]


def build_dataset(cfg, transform=None, target_transform=None, is_train=True):
    dataset_name = cfg.DATASET.NAME
    assert dataset_name in __supported_dataset__, f"{dataset_name} do not support"

    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT

    # Data loading code
    if dataset_name == 'GeneralDataset':
        dataset = GeneralDataset(
            data_root, transform=transform, target_transform=target_transform
        )
    elif dataset_name == 'GeneralDatasetV2':
        dataset = GeneralDatasetV2(
            data_root, transform=transform, target_transform=target_transform
        )
    elif dataset_name == 'MPDataset':
        num_gpus = cfg.NUM_GPUS
        rank_id = cfg.RANK_ID
        epoch = cfg.TRAIN.START_EPOCH

        dataset = MPDataset(
            data_root, transform=transform, target_transform=target_transform,
            shuffle=is_train, num_gpus=num_gpus, rank_id=rank_id, epoch=epoch
        )
    else:
        raise ValueError(f"{dataset_name} do not support")

    return dataset
