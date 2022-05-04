# -*- coding: utf-8 -*-

"""
@date: 2022/5/4 下午3:36
@file: ghostnet.py
@author: zj
@description: 
"""

import timm


def main():
    m = timm.models.__dict__['ghostnet_130']()
    print(m)

    import torch
    data = torch.randn(1, 3, 224, 224)
    print('data:', data.shape)
    res = m.forward_features(data)
    print('res:', res.shape)


if __name__ == '__main__':
    main()
