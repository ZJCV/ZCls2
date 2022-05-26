# -*- coding: utf-8 -*-

"""
@date: 2022/4/16 下午7:55
@file: mix_dataset.py
@author: zj
@description: CCCF is a custom mixed classification dataset
"""
from typing import Optional, Tuple, Any, Callable, List

import os
from PIL import Image

from torch.utils.data import Dataset
from zcls2.config.key_word import KEY_SEP

__all__ = ['CCCF']


def load_txt(txt_path: str, delimiter: str = ',,') -> List:
    assert os.path.isfile(txt_path), txt_path

    data_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_list = line.strip().split(delimiter)
            data_list.append(tmp_list)

    return data_list


def load_classes(class_path: str, delimiter=' ') -> List:
    assert os.path.isfile(class_path), class_path

    class_list = []
    with open(class_path, 'r', encoding='utf-8') as f:
        for line in f:
            class_name = line.strip().split(delimiter)[0]
            class_list.append(class_name)

    return class_list


class CCCF(Dataset):
    """
    CCCF is a custom mixed classification dataset, including

    1. CIFAR100: https://paperswithcode.com/dataset/cifar-100
    2. CUB-200-2011: https://paperswithcode.com/dataset/cub-200-2011
    3. Caltech-101: https://paperswithcode.com/dataset/caltech-101
    4. Food-101: https://paperswithcode.com/dataset/food-101

    The classes num = 100 + 200 + 101 + 101 = 502
    """

    def __init__(self, root: str,
                 train: Optional[bool] = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        assert os.path.isdir(root), root

        class_path = os.path.join(root, 'classes.txt')
        assert os.path.isfile(class_path), class_path
        train_path = os.path.join(root, 'train.txt')
        assert os.path.isfile(train_path), train_path
        test_path = os.path.join(root, 'test.txt')
        assert os.path.isfile(test_path), test_path

        classes = load_classes(class_path, delimiter=' ')
        data_list = load_txt(train_path, delimiter=KEY_SEP) if train else \
            load_txt(test_path, delimiter=KEY_SEP)

        self.classes = classes
        self.data = [os.path.join(root, str(img_path)) for img_path, target in data_list]
        self.targets = [int(target) for img_path, target in data_list]

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return self.__class__.__name__ + ' (' + self.root + ')'
