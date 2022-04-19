# -*- coding: utf-8 -*-

"""
@date: 2022/4/16 下午7:55
@file: mix_dataset.py
@author: zj
@description: Custom mixed classification data set, including CIFAR100, FashionMNIST and CUB_2012_2011.
The classes num = 100 + 10 + 10 = 120
"""

import os

from typing import Optional, Tuple, Any, Callable

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, FashionMNIST, SVHN

__all__ = ['MixDataset']


class MixDataset(Dataset):

    def __init__(self, root: str,
                 train: Optional[bool] = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        assert os.path.isdir(root), root

        cifar100_root = os.path.join(root, 'cifar100')
        self.cifar100_dataset = CIFAR100(cifar100_root, train=train,
                                         transform=transform, target_transform=target_transform, download=True)

        fashionmnist_root = os.path.join(root, 'fashionmnist')
        self.fashionmnist_dataset = FashionMNIST(fashionmnist_root, train=train,
                                                 transform=transform, target_transform=target_transform, download=True)

        voc_root = os.path.join(root, 'svhn')
        self.svhn_dataset = SVHN(voc_root, split='train' if train else 'test',
                                 transform=transform, target_transform=target_transform, download=True)

        self.length = len(self.cifar100_dataset) + len(self.fashionmnist_dataset) + len(self.svhn_dataset)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        if index < self.cifar100_dataset.__len__():
            img, target = self.cifar100_dataset.__getitem__(index)
        else:
            index = index - len(self.cifar100_dataset)
            if index < len(self.fashionmnist_dataset):
                img, target = self.fashionmnist_dataset.__getitem__(index)
                target += 100
            else:
                index = index - len(self.fashionmnist_dataset)
                img, target = self.svhn_dataset.__getitem__(index)
                target += 10

        return img, int(target)

    def __len__(self) -> int:
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
