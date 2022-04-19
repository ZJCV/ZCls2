# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 上午11:04
@file: general_dataset.py
@author: zj
@description: Custom Dataset
"""

import os

from typing import Optional, Tuple, Any, Callable

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from zcls2.config.key_word import KEY_DATASET, KEY_CLASSES, KEY_SEP

__all__ = ['GeneralDatasetV2']


def get_base_info(cls_path: str, data_path: str) -> Tuple[list, list, list]:
    assert os.path.isfile(cls_path), cls_path
    classes = list()
    with open(cls_path, 'r') as f:
        for idx, line in enumerate(f):
            classes.append(line.strip())

    assert os.path.isfile(data_path), data_path
    length = 0
    with open(data_path, 'r') as f:
        for _ in f:
            length += 1

    data_list = ["" for _ in range(length)]
    target_list = ["" for _ in range(length)]
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            img_path, target = line.strip().split(KEY_SEP)[:2]

            data_list[idx] = img_path
            target_list[idx] = target

    return classes, data_list, target_list


class GeneralDatasetV2(Dataset):

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        assert os.path.isdir(root), root
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.data_path = os.path.join(self.root, KEY_DATASET)
        self.cls_path = os.path.join(self.root, KEY_CLASSES)
        classes, data_list, target_list = get_base_info(self.cls_path, self.data_path)

        self.classes = classes
        self.data_list = data_list
        self.target_list = target_list
        self.length = len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = self.data_list[index]
        target = self.target_list[index]

        image = default_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, int(target)

    def __len__(self) -> int:
        return self.length

    def get_classes(self) -> list:
        return self.classes

    def __repr__(self) -> str:
        return self.__class__.__name__ + ' (' + self.root + ')'
