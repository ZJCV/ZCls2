# -*- coding: utf-8 -*-

"""
@date: 2022/4/11 下午7:59
@file: square_pad.py
@author: zj
@description: 
"""

import torch

import numpy as np
from PIL import Image

from zcls2.config import cfg
from zcls2.data.transform.build import build_transform


def square_pad():
    config_file = 'configs/square_pad.yaml'
    cfg.merge_from_file(config_file)
    # print(cfg.TRANSFORM)

    tran, _ = build_transform(cfg, is_train=True)
    print(tran)

    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))
    res = tran(data)
    print(type(res))
    print(res.size)


if __name__ == '__main__':
    square_pad()
