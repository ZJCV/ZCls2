# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:30
@file: parser.py
@author: zj
@description:
"""

import os

import argparse
from argparse import Namespace

from yacs.config import CfgNode


def parse() -> Namespace:
    parser = argparse.ArgumentParser(description='ZCls2 Training with Pytorch')
    parser.add_argument('-cfg',
                        "--config",
                        type=str,
                        default="",
                        metavar="CONFIG",
                        help="path to config file")

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)

    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()

    return args


def load_cfg(args: Namespace, cfg: CfgNode):
    cfg.DISTRIBUTED = args.distributed
    cfg.RANK_ID = args.gpu
    cfg.NUM_GPUS = args.world_size

    cfg.DETERMINISTIC = args.deterministic
    cfg.RNG_SEED = args.local_rank

    cfg.EVALUATE = args.evaluate

    cfg.CHANNELS_LAST = args.channels_last
