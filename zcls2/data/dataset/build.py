# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:46
@file: build.py
@author: zj
@description: Dataset builder
"""

from typing import Optional
from yacs.config import CfgNode

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

from . import general_dataset, mp_dataset, cccf

__all__ = ['build_dataset']


def build_dataset(cfg: CfgNode,
                  transform: Optional[transforms.Compose] = None,
                  target_transform: Optional[transforms.Compose] = None,
                  is_train: Optional[bool] = True) -> Dataset:
    dataset_name = cfg.DATASET.NAME
    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT

    # Data loading code
    if dataset_name == "ImageFolder":
        dataset = ImageFolder(
            data_root, transform=transform, target_transform=target_transform
        )
    elif dataset_name in general_dataset.__all__:
        dataset = general_dataset.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform
        )
    elif dataset_name in mp_dataset.__all__:
        num_gpus = cfg.NUM_GPUS
        rank_id = cfg.RANK_ID
        epoch = cfg.TRAIN.START_EPOCH

        dataset = mp_dataset.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform,
            shuffle=is_train, num_gpus=num_gpus, rank_id=rank_id, epoch=epoch
        )
    elif dataset_name in cccf.__all__:
        dataset = cccf.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, train=is_train
        )
    else:
        raise ValueError(f"{dataset_name} do not support")

    return dataset
