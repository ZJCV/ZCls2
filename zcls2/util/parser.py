# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:30
@file: parser.py
@author: zj
@description: 
"""

import os
import argparse

from ..model.model.build import __supported_model__
from ..data.dataset.build import __supported_dataset__


def parse():
    parser = argparse.ArgumentParser(description='ZCls2 Training with Pytorch')
    parser.add_argument('-cfg',
                        "--config",
                        type=str,
                        default="",
                        metavar="CONFIG",
                        help="path to config file")

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--output-dir', '-o', default='outputs', type=str,
                        metavar='OUTPUT_DIR', help='output path (default: outputs)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()

    return args
