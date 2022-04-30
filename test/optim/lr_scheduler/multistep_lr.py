# -*- coding: utf-8 -*-

"""
@date: 2022/4/14 上午11:25
@file: cosine_annealing_lr.py
@author: zj
@description: 
"""

import torch

from zcls2.config import get_cfg_defaults
from zcls2.model.model.build import build_model
from zcls2.optim.optimizer.build import build_optimizer
from zcls2.optim.lr_scheduler.build import build_lr_scheduler


def ca():
    cfg = get_cfg_defaults()
    config_file = 'configs/mslr.yaml'
    cfg.merge_from_file(config_file)
    # print(cfg)

    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    max_epoch = cfg.TRAIN.MAX_EPOCH
    for i in range(max_epoch):
        print(i, optimizer.param_groups[0]['lr'])
        optimizer.step()
        lr_scheduler.step()


def cav2():
    cfg = get_cfg_defaults()
    config_file = 'configs/mslrv2.yaml'
    cfg.merge_from_file(config_file)
    # print(cfg)

    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    max_epoch = cfg.TRAIN.MAX_EPOCH
    for i in range(max_epoch):
        print(i, optimizer.param_groups[0]['lr'])
        optimizer.step()
        lr_scheduler.step()


if __name__ == '__main__':
    # ca()
    cav2()
