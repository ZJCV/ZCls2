# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:46
@file: build.py
@author: zj
@description: 
"""

from typing import Optional
from yacs.config import CfgNode

from torch.utils.data import Dataset
import torchvision.transforms.transforms as transforms

from . import general_dataset, general_dataset_v2, mp_dataset, mix_dataset

__all__ = general_dataset.__all__ \
          + general_dataset_v2.__all__ \
          + mp_dataset.__all__ \
          + mix_dataset.__all__


def build_dataset(cfg: CfgNode,
                  transform: Optional[transforms.Compose] = None,
                  target_transform: Optional[transforms.Compose] = None,
                  is_train: Optional[bool] = True) -> Dataset:
    dataset_name = cfg.DATASET.NAME
    assert dataset_name in __all__, f"{dataset_name} do not support"

    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT

    # Data loading code
    if dataset_name in general_dataset.__all__:
        dataset = general_dataset.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform
        )
    elif dataset_name in general_dataset_v2.__all__:
        dataset = general_dataset_v2.__dict__[dataset_name](
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
    elif dataset_name in mix_dataset.__all__:
        dataset = mix_dataset.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, train=is_train
        )
    else:
        raise ValueError(f"{dataset_name} do not support")

    return dataset
