# -*- coding: utf-8 -*-

"""
@date: 2022/4/30 下午2:22
@file: efficientnet.py
@author: zj
@description: Custom EfficientNet, derived from torchvision
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
Derived from https://github.com/lukemelas/EfficientNet-PyTorch/blob/7e8b0d312162f335785fb5dcfa1df29a75a1783a/efficientnet_pytorch/utils.py#L466
"""

from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Dict

from torchvision.models import efficientnet
from torchvision.models import EfficientNet as TEfficientNet
from torchvision.models.efficientnet import MBConvConfig

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from zcls2.config.key_word import KEY_OUTPUT

__all__ = ["EfficientNet",
           "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
           "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]


class EfficientNet(TEfficientNet):

    def __init__(self, inverted_residual_setting: List[MBConvConfig], dropout: float,
                 stochastic_depth_prob: float = 0.2, num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, **kwargs: Any) -> None:
        super().__init__(inverted_residual_setting, dropout, stochastic_depth_prob, num_classes, block, norm_layer,
                         **kwargs)

    def forward(self, x: Tensor) -> Dict:
        x = super().forward(x)

        return {
            KEY_OUTPUT: x
        }


def _efficientnet_conf(width_mult: float, depth_mult: float, **kwargs: Any) -> List[MBConvConfig]:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    return inverted_residual_setting


def _efficientnet_model(
        arch: str,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> EfficientNet:
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(efficientnet.model_urls[arch],
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


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.0, **kwargs)
    return _efficientnet_model("efficientnet_b0", inverted_residual_setting, 0.2, pretrained, progress, **kwargs)


def efficientnet_b1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.1, **kwargs)
    return _efficientnet_model("efficientnet_b1", inverted_residual_setting, 0.2, pretrained, progress, **kwargs)


def efficientnet_b2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.1, depth_mult=1.2, **kwargs)
    return _efficientnet_model("efficientnet_b2", inverted_residual_setting, 0.3, pretrained, progress, **kwargs)


def efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.2, depth_mult=1.4, **kwargs)
    return _efficientnet_model("efficientnet_b3", inverted_residual_setting, 0.3, pretrained, progress, **kwargs)


def efficientnet_b4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.4, depth_mult=1.8, **kwargs)
    return _efficientnet_model("efficientnet_b4", inverted_residual_setting, 0.4, pretrained, progress, **kwargs)


def efficientnet_b5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.6, depth_mult=2.2, **kwargs)
    return _efficientnet_model("efficientnet_b5", inverted_residual_setting, 0.4, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)


def efficientnet_b6(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.8, depth_mult=2.6, **kwargs)
    return _efficientnet_model("efficientnet_b6", inverted_residual_setting, 0.5, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)


def efficientnet_b7(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=2.0, depth_mult=3.1, **kwargs)
    return _efficientnet_model("efficientnet_b7", inverted_residual_setting, 0.5, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)
