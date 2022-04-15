# -*- coding: utf-8 -*-

"""
@date: 2022/4/10 上午10:51
@file: mobilenet.py
@author: zj
@description: 
"""

import torch.nn as nn
import torchvision.models as models

from typing import Optional

from .util import create_linear

__supported_model__ = [
    'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'
]


def get_mobilenet(pretrained: Optional[bool] = False,
                  num_classes: Optional[int] = 1000,
                  arch: Optional[str] = 'mobilenet_v2') -> nn.Module:
    assert arch in __supported_model__, f"{arch} not in {__supported_model__}"

    if pretrained == True:
        model = models.__dict__[arch](pretrained=True, num_classes=1000)
        if num_classes != 1000:
            if arch == 'mobilenet_v2':
                old_fc = model.classifier[1]
                assert isinstance(old_fc, nn.Linear)

                in_features = old_fc.in_features
                new_fc = create_linear(in_features, num_classes, bias=old_fc.bias is not None)

                model.classifier[1] = None
                model.classifier[1] = new_fc
            else:
                old_fc = model.classifier[3]
                assert isinstance(old_fc, nn.Linear)

                in_features = old_fc.in_features
                new_fc = create_linear(in_features, num_classes, bias=old_fc.bias is not None)

                model.classifier[3] = None
                model.classifier[3] = new_fc
    else:
        model = models.__dict__[arch](pretrained=False, num_classes=num_classes)

    return model
