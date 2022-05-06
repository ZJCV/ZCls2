# -*- coding: utf-8 -*-

"""
@date: 2022/4/10 上午10:51
@file: mobilenet.py
@author: zj
@description: Custom MobileNetV2/V3, derived from torchvision
"""
from typing import Any, List, Optional, Callable, Dict

from torch import nn, Tensor
from torchvision.models.mobilenet import MobileNetV2 as TMobileNetV2
from torchvision.models.mobilenet import MobileNetV3 as TMobileNetV3
from torchvision.models import mobilenetv2
from torchvision.models import mobilenetv3

# See vision/torchvision/_internally_replaced_utils.py
# https://github.com/pytorch/vision/blob/b50ffef5f85029b1440ac155ca1e6d95c55520aa/torchvision/_internally_replaced_utils.py
from torchvision.models.mobilenetv3 import InvertedResidualConfig

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from zcls2.config.key_word import KEY_OUTPUT

__all__ = [
    "MobileNetV2", "MobileNetV3",
    'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'
]


class MobileNetV2(TMobileNetV2):

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0,
                 inverted_residual_setting: Optional[List[List[int]]] = None, round_nearest: int = 8,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(num_classes, width_mult, inverted_residual_setting, round_nearest, block, norm_layer)

    def forward(self, x: Tensor) -> Dict:
        res = super().forward(x)
        return {
            KEY_OUTPUT: res
        }


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(mobilenetv2.model_urls['mobilenet_v2'],
                                              progress=progress)

        # If the number of model outputs is different from the model setting,
        # the corresponding pretraining model weight will not be loaded
        assert isinstance(model.classifier[1], nn.Linear)
        if model.classifier[1].out_features != 1000:
            state_dict.pop('classifier.1.weight')
            state_dict.pop('classifier.1.bias')

        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == {'classifier.1.weight', 'classifier.1.bias'}, \
            f'Missing keys when loading pretrained weights: {ret.missing_keys}'
    return model


class MobileNetV3(TMobileNetV3):

    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int,
                 num_classes: int = 1000, block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, **kwargs: Any) -> None:
        super().__init__(inverted_residual_setting, last_channel, num_classes, block, norm_layer, **kwargs)

    def forward(self, x: Tensor) -> Dict:
        res = super().forward(x)
        return {
            KEY_OUTPUT: res
        }


def _mobilenet_v3_model(
        arch: str,
        inverted_residual_setting: List[mobilenetv3.InvertedResidualConfig],
        last_channel: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        if mobilenetv3.model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(mobilenetv3.model_urls[arch], progress=progress)

        # If the number of model outputs is different from the model setting,
        # the corresponding pretraining model weight will not be loaded
        assert isinstance(model.classifier[3], nn.Linear)
        if model.classifier[3].out_features != 1000:
            state_dict.pop('classifier.3.weight')
            state_dict.pop('classifier.3.bias')

        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == {'classifier.3.weight', 'classifier.3.bias'}, \
            f'Missing keys when loading pretrained weights: {ret.missing_keys}'
    return model


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)
