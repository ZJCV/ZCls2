# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:51
@file: transform.py
@author: zj
@description: Transform settings
"""

from yacs.config import CfgNode as CN


def add_config(_C: CN) -> None:
    # ---------------------------------------------------------------------------- #
    # Transform
    # ---------------------------------------------------------------------------- #
    _C.TRANSFORM = CN()
    # _C.TRANSFORM.TRAIN_METHODS = ('Resize', 'CenterCrop', 'ToTensor', 'Normalize')
    # _C.TRANSFORM.TEST_METHODS = ('Resize', 'CenterCrop', 'ToTensor', 'Normalize')
    _C.TRANSFORM.TRAIN_METHODS = ('RandomResizedCrop', 'RandomHorizontalFlip')
    _C.TRANSFORM.TEST_METHODS = ('Resize', 'CenterCrop')

    # ConvertImageDtype(dtype)
    # dtype: ['uint8', 'float32']
    _C.TRANSFORM.ConvertImageDtype = 'uint8'

    # Normalize(mean / max_value, std / max_value, inplace=False)
    # Default using Mean and STD calculated using Imagenet
    # Args:
    #     mean (sequence): Sequence of means for each channel.
    #     std (sequence): Sequence of standard deviations for each channel.
    #     inplace(bool,optional): Bool to make this operation in-place.
    #     max_value (int): Max value in pixel value
    _C.TRANSFORM.NORMALIZE = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), False, 1.0)

    # ---------------------------------------------------------------------------- #
    # Augment transform
    # ---------------------------------------------------------------------------- #
    # AutoAugment(policy, interpolation)
    # policy: ['IMAGENET', 'CIFAR10', 'SVHN']
    # interpolation: ['NEAREST', 'BILINEAR', 'BICUBIC']
    _C.TRANSFORM.AutoAugment = ("IMAGENET", "NEAREST")

    # ---------------------------------------------------------------------------- #
    # Geometric transform
    # ---------------------------------------------------------------------------- #
    # Desired output size of the crop.
    # If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
    # If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    _C.TRANSFORM.TRAIN_CROP = (224,)
    _C.TRANSFORM.TEST_CROP = (224,)

    # RandomHorizontalFlip(p=0.5)
    # Args:
    #     p (float): probability of the image being flipped. Default value is 0.5
    _C.TRANSFORM.RandomHorizontalFlip = 0.5

    # RandomVerticalFlip(p=0.5)
    # Args:
    #     p (float): probability of the image being flipped. Default value is 0.5
    _C.TRANSFORM.RandomVerticalFlip = 0.5

    # RandomRotate(degrees, interpolation=InterpolationMode.NEAREST, expand=False)
    # interpolation: ['NEAREST', 'BILINEAR', 'BICUBIC']
    # Args:
    #    degrees (sequence or number): Range of degrees to select from.
    #         If degrees is a number instead of sequence like (min, max), the range of degrees
    #         will be (-degrees, +degrees).
    #     interpolation (InterpolationMode): Desired interpolation enum defined by
    #         :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
    #         If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
    #         For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
    #     expand (bool, optional): Optional expansion flag.
    #         If true, expands the output to make it large enough to hold the entire rotated image.
    #         If false or omitted, make the output image the same size as the input image.
    #         Note that the expand flag assumes rotation around the center and no translation.
    _C.TRANSFORM.RandomRotate = (0, "NEAREST", False)

    # If size is a sequence like (h, w), output size will be matched to this.
    # If size is an int, smaller edge of the image will be matched to this number.
    # i.e, if height > width, then image will be rescaled to (size * height / width, size).
    _C.TRANSFORM.TRAIN_RESIZE = (224,)
    _C.TRANSFORM.TEST_RESIZE = (256,)

    # Desired output size of the crop.
    # If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
    # If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    _C.TRANSFORM.TRAIN_RESIZE_CROP = (224,)
    _C.TRANSFORM.TEST_RESIZE_CROP = (224,)

    # ---------------------------------------------------------------------------- #
    # Color transform
    # ---------------------------------------------------------------------------- #
    # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    # brightness (float or tuple of float (min, max)): How much to jitter brightness.
    #     brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
    #     or the given [min, max]. Should be non negative numbers.
    # contrast (float or tuple of float (min, max)): How much to jitter contrast.
    #     contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
    #     or the given [min, max]. Should be non negative numbers.
    # saturation (float or tuple of float (min, max)): How much to jitter saturation.
    #     saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
    #     or the given [min, max]. Should be non negative numbers.
    # hue (float or tuple of float (min, max)): How much to jitter hue.
    #     hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
    #     Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    _C.TRANSFORM.ColorJitter = (0, 0, 0, 0)

    # RandomAutocontrast(p=0.5)
    # p (float): probability of the image being autocontrasted. Default value is 0.5
    _C.TRANSFORM.RandomAutocontrast = 0.5

    # RandomAdjustSharpness(sharpness_factor, p=0.5)
    # sharpness_factor (float):  How much to adjust the sharpness. Can be
    #    any non negative number. 0 gives a blurred image, 1 gives the
    #    original image while 2 increases the sharpness by a factor of 2.
    # p (float): probability of the image being color inverted. Default value is 0.5
    _C.TRANSFORM.RandomAdjustSharpness = (1, 0.5)

    # RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    # Args:
    #      p: probability that the random erasing operation will be performed.
    #      scale: range of proportion of erased area against input image.
    #      ratio: range of aspect ratio of erased area.
    #      value: erasing value. Default is 0. If a single int, it is used to
    #         erase all pixels. If a tuple of length 3, it is used to erase
    #         R, G, B channels respectively.
    #         If a str of 'random', erasing each pixel with random values.
    #      inplace: boolean to make this transform inplace. Default set to False.
    _C.TRANSFORM.RandomErasing = (0.5, (0.02, 0.33), (0.3, 3.3), 0, False)

    # RandomPosterize(bits, p=0.5)
    # Args:
    #     bits (int): number of bits to keep for each channel (0-8)
    #     p (float): probability of the image being color inverted. Default value is 0.5
    _C.TRANSFORM.RandomPosterize = (8, 0.5)

    # Mixup + Cutmix
    _C.TRANSFORM.MIXUP = CN()
    # Enable Mixup and Cutmix (default: False)
    _C.TRANSFORM.MIXUP.MIXUP_ENABLED = False
    # Mixup alpha, mixup enabled if > 0.
    _C.TRANSFORM.MIXUP.MIXUP_ALPHA = 0.8
    # Cutmix alpha, cutmix enabled if > 0.
    _C.TRANSFORM.MIXUP.CUTMIX_ALPHA = 1.0
    # Cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
    _C.TRANSFORM.MIXUP.CUTMIX_MINMAX = None
    # Probability of performing mixup or cutmix when either/both is enabled
    _C.TRANSFORM.MIXUP.MIXUP_PROB = 1.0
    # Probability of switching to cutmix when both mixup and cutmix enabled
    _C.TRANSFORM.MIXUP.MIXUP_SWITCH_PROB = 0.5
    # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
    _C.TRANSFORM.MIXUP.MIXUP_MODE = 'batch'

