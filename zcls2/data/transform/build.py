# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:50
@file: build.py
@author: zj
@description: 
"""

import torch

import torchvision.transforms.transforms as transforms
import torchvision.transforms.autoaugment as autoaugment
import torchvision.transforms.functional as F

from .square_pad import SquarePad
from .resize import Resize

__supported_transform__ = [
    # Normal transform
    'ConvertImageDtype',
    'Normalize',
    'ToPILImage',
    'ToTensor',
    # Color transform
    'ColorJitter',
    'Grayscale',
    'RandomAutocontrast',
    'RandomAdjustSharpness',
    'RandomErasing',
    'RandomPosterize',
    # Geometric transform
    'CenterCrop',
    'RandomCrop',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'RandomResizedCrop',
    'Resize',
    # augment
    'AutoAugment',
    'RandAugment',
    # custom
    'OpenCVResize',
    'SquarePad',
]


def parse_transform(cfg, is_train=True):
    methods = cfg.TRANSFORM.TRAIN_METHODS if is_train else cfg.TRANSFORM.TEST_METHODS
    assert isinstance(methods, tuple)

    aug_list = list()
    for method in methods:
        assert method in __supported_transform__

        # Normal transform
        if method == 'ConvertImageDtype':
            assert cfg.TRANSFORM.ConvertImageDtype in ['uint8', 'float32']
            dtype = torch.uint8 if cfg.TRANSFORM.ConvertImageDtype == 'uint8' else torch.float32
            tf = transforms.ConvertImageDtype(dtype)
        elif method == 'Normalize':
            mean, std, inplace = cfg.TRANSFORM.NORMALIZE
            assert len(mean) == len(std)
            tf = transforms.Normalize(mean, std, inplace=inplace)
        elif method == 'ToPILImage':
            tf = transforms.ToPILImage()
        elif method == 'ToTensor':
            tf = transforms.ToTensor()
        # Color transform
        elif method == 'ColorJitter':
            brightness, contrast, saturation, hue = cfg.TRANSFORM.ColorJitter
            tf = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        elif method == 'Grayscale':
            tf = transforms.Grayscale()
        elif method == 'RandomAutocontrast':
            p = cfg.TRANSFORM.RandomAutocontrast
            tf = transforms.RandomAutocontrast(p=p)
        elif method == 'RandomAdjustSharpness':
            sharpness_factor, p = cfg.TRANSFORM.RandomAdjustSharpness
            tf = transforms.RandomAdjustSharpness(sharpness_factor, p=p)
        elif method == 'RandomErasing':
            p, scale, ratio, value, inplace = cfg.TRANSFORM.RandomErasing
            tf = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)
        elif method == 'RandomPosterize':
            bits, p = cfg.TRANSFORM.RandomPosterize
            tf = transforms.RandomPosterize(bits, p=p)
        # Geometric transform
        elif method == 'CenterCrop':
            size = cfg.TRANSFORM.TRAIN_CROP if is_train else cfg.TRANSFORM.TEST_CROP
            tf = transforms.CenterCrop(size)
        elif method == 'RandomCrop':
            size = cfg.TRANSFORM.TRAIN_CROP if is_train else cfg.TRANSFORM.TEST_CROP
            tf = transforms.RandomCrop(size)
        elif method == 'RandomHorizontalFlip':
            p = cfg.TRANSFORM.RandomHorizontalFlip
            tf = transforms.RandomHorizontalFlip(p=p)
        elif method == 'RandomVerticalFlip':
            p = cfg.TRANSFORM.RandomVerticalFlip
            tf = transforms.RandomVerticalFlip(p=p)
        elif method == 'RandomRotation':
            degrees, interpolation, expand = cfg.TRANSFORM.RandomRotate
            assert interpolation in ['NEAREST', 'BILINEAR', 'BICUBIC']
            tf = transforms.RandomRotation(degrees, interpolation=F.InterpolationMode[interpolation], expand=expand)
        elif method == 'RandomResizedCrop':
            size = cfg.TRANSFORM.TRAIN_RESIZE_CROP if is_train else cfg.TRANSFORM.TEST_RESIZE_CROP
            tf = transforms.RandomResizedCrop(size)
        elif method == 'Resize':
            size = cfg.TRANSFORM.TRAIN_RESIZE if is_train else cfg.TRANSFORM.TEST_RESIZE
            tf = transforms.Resize(size)
        # Custom methods
        elif method == 'SquarePad':
            tf = SquarePad()
        elif method == 'OpenCVResize':
            size = cfg.TRANSFORM.TRAIN_RESIZE if is_train else cfg.TRANSFORM.TEST_RESIZE
            tf = Resize(size)
        # Augment methods
        elif method == 'AutoAugment':
            policy, interpolation = cfg.TRANSFORM.AutoAugment
            assert policy in ['IMAGENET', 'CIFAR10', 'SVHN']
            assert interpolation in ['NEAREST', 'BILINEAR', 'BICUBIC']

            tf = autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy[policy], F.InterpolationMode[interpolation])
        elif method == 'RandAugment':
            tf = autoaugment.RandAugment()
        else:
            raise ValueError(f'{method} does not exists')

        aug_list.append(tf)

    return transforms.Compose(aug_list)


def parse_target_transform():
    return None


def build_transform(cfg, is_train=True):
    return parse_transform(cfg, is_train), parse_target_transform()
