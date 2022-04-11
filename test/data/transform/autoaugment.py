# -*- coding: utf-8 -*-

"""
@date: 2022/4/11 下午5:05
@file: autoaugment.py
@author: zj
@description: 
"""

import torch
import numpy as np
from PIL import Image

from zcls2.config import cfg
from zcls2.data.transform.build import build_transform


def augment():
    # print(cfg.TRANSFORM)
    # print(cfg.TRANSFORM.AutoAugment)
    cfg.TRANSFORM.TRAIN_METHODS = ('AutoAugment',)

    tran, _ = build_transform(cfg, is_train=True)
    print(tran)
    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))

    res = tran(data)
    print(res.size)


if __name__ == '__main__':
    augment()
