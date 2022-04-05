# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 下午3:10
@file: resize.py
@author: zj
@description: 
"""

import cv2

from PIL import Image
import numpy as np
import torch.nn as nn


def get_hw(img: np.ndarray, size: int, mode: int):
    assert mode in [0, 1]
    h, w = img.shape[:2]

    short, long = (w, h) if w <= h else (h, w)
    if mode == 0:
        new_short, new_long = size, int(size * long / short)
    else:
        new_short, new_long = int(size * short / long), size

    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return new_h, new_w


class Resize(nn.Module):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        super().__init__()
        if isinstance(size, tuple):
            size = list(size)

        if isinstance(size, int):
            size = [size]
        elif isinstance(size, list):
            assert len(size) == 2
        else:
            raise ValueError(f'{size} do not support')
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        # 将PIL图像转换成numpy
        if isinstance(img, Image.Image):
            img = np.array(img)

        if len(self.size) == 2:
            new_h, new_w = self.size[:2]
        else:
            new_h, new_w = get_hw(img, self.size[0], 0)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
        return Image.fromarray(new_img)

    def __repr__(self):
        interpolate_str = str(self.interpolation)
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


if __name__ == '__main__':
    img = Image.open('/home/zj/opencv/test/make_test/lena.jpg')
    # img.show()

    m = Resize(224)
    res = m.forward(img)

    res.show()
