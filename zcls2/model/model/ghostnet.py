# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午5:20
@file: ghostnet.py
@author: zj
@description: 
"""
from typing import Optional, Dict

from torch import Tensor, nn
import torch.nn.functional as F

import timm.models as models
from timm.models import GhostNet

from zcls2.config.key_word import KEY_OUTPUT
from .util import create_linear

__supported_model__ = ['ghostnet_050', 'ghostnet_100', 'ghostnet_130']


def forward(self: GhostNet, x: Tensor) -> Dict:
    x = self.forward_features(x)
    x = self.flatten(x)
    if self.dropout > 0.:
        x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.classifier(x)
    return {
        KEY_OUTPUT: x
    }


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

    model.forward = forward
    return model


if __name__ == '__main__':
    model = get_ghostnet(num_classes=501, arch='ghostnet_130', pretrained=True)
    print(model)
