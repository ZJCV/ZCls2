# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午5:20
@file: ghostnet.py
@author: zj
@description: 
"""

import torch.nn as nn
from typing import Optional
import timm.models as models

from .util import create_linear

__supported_model__ = ['ghostnet_050', 'ghostnet_100', 'ghostnet_130']


def get_ghostnet(pretrained: Optional[bool] = False,
                 num_classes: Optional[int] = 1000,
                 arch: Optional[str] = 'ghostnet_050') -> nn.Module:
    assert arch in __supported_model__, f"{arch} not in {__supported_model__}"

    if pretrained == True:
        model = models.__dict__[arch](pretrained=True, num_classes=1000)
        if num_classes != 1000:
            old_fc = model.classifier
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = create_linear(in_features, num_classes, bias=old_fc.bias is not None)

            model.classifier = None
            model.classifier = new_fc
    else:
        model = models.__dict__[arch](pretrained=False, num_classes=num_classes)

    return model


if __name__ == '__main__':
    model = get_ghostnet(num_classes=501, arch='ghostnet_130', pretrained=True)
    print(model)
