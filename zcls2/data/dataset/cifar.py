# -*- coding: utf-8 -*-

"""
@date: 2020/11/10 下午5:02
@file: cifar.py
@author: zj
@description: 
"""

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100


class CIFAR(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, is_cifar100=True):
        if is_cifar100:
            self.data_set = CIFAR100(root, train=train, transform=transform, target_transform=target_transform,
                                     download=True)
        else:
            self.data_set = CIFAR10(root, train=train, transform=transform, target_transform=target_transform,
                                    download=True)
        self.classes = self.data_set.classes
        self.root = root

    def __getitem__(self, index: int):
        return self.data_set.__getitem__(index)

    def __len__(self) -> int:
        return self.data_set.__len__()

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
