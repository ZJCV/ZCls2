# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 上午11:04
@file: general_dataset.py
@author: zj
@description: like torchvision.datasets.ImageFolder
"""

from typing import Optional, Tuple, Any, Callable

from torch.utils.data import Dataset
import torchvision.datasets as datasets

__all__ = ['GeneralDataset']


class GeneralDataset(Dataset):

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.data_set = datasets.ImageFolder(root, transform=transform, target_transform=target_transform)
        self.classes = self.data_set.classes
        self.root = root

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = self.data_set.__getitem__(index)

        return image, target

    def __len__(self) -> int:
        return self.data_set.__len__()

    def get_classes(self) -> list:
        return self.classes

    def __repr__(self) -> str:
        return self.__class__.__name__ + ' (' + self.root + ')'
