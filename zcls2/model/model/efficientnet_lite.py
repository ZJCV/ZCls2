# -*- coding: utf-8 -*-

"""
@date: 2022/4/30 下午10:52
@file: efficientnet_lite.py
@author: zj
@description: Custom EfficientNetLite, derived from [ RangiLyu/effnet_lite](https://github.com/RangiLyu/EfficientNet-Lite/blob/main/efficientnet_lite.py)
"""
from typing import Dict

from zcls2.config.key_word import KEY_OUTPUT

from .effnet_lite.efficientnet_lite import efficientnet_lite_params
from .effnet_lite.efficientnet_lite import EfficientNetLite as TEfficientNetLite

__all__ = ['EfficientNetLite',
           'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2', 'efficientnet_lite3',
           'efficientnet_lite4']


class EfficientNetLite(TEfficientNetLite):

    def __init__(self, widthi_multiplier, depth_multiplier, num_classes, drop_connect_rate, dropout_rate):
        super().__init__(widthi_multiplier, depth_multiplier, num_classes, drop_connect_rate, dropout_rate)

    def forward(self, x) -> Dict:
        x = super().forward(x)
        return {
            KEY_OUTPUT: x
        }


def efficientnet_lite0(num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = efficientnet_lite_params["efficientnet_lite0"]
    model = EfficientNetLite(width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate)
    return model


def efficientnet_lite1(num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = efficientnet_lite_params["efficientnet_lite1"]
    model = EfficientNetLite(width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate)
    return model


def efficientnet_lite2(num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = efficientnet_lite_params["efficientnet_lite2"]
    model = EfficientNetLite(width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate)
    return model


def efficientnet_lite3(num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = efficientnet_lite_params["efficientnet_lite3"]
    model = EfficientNetLite(width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate)
    return model


def efficientnet_lite4(num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = efficientnet_lite_params["efficientnet_lite4"]
    model = EfficientNetLite(width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate)
    return model
