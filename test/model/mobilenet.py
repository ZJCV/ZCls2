# -*- coding: utf-8 -*-

"""
@date: 2022/4/22 下午4:06
@file: mobilenet.py
@author: zj
@description: 
"""

from zcls2.config import get_cfg_defaults
from zcls2.model.model.build import build_model


def mbv2():
    cfg = get_cfg_defaults()
    CONFIG_FILE = 'configs/mbv2.yaml'
    cfg.merge_from_file(CONFIG_FILE)
    # print(cfg)

    model = build_model(cfg)
    print(model)


def mbv3():
    cfg = get_cfg_defaults()
    CONFIG_FILE = 'configs/mbv3.yaml'
    cfg.merge_from_file(CONFIG_FILE)
    # print(cfg)

    model = build_model(cfg)
    print(model)


if __name__ == '__main__':
    mbv2()
    mbv3()
