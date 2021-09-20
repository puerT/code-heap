#!/usr/bin/env python3

import numpy as np

from .bicubic import bicubic
from .bilinear import bilinear
from .nearest import nearest


def grid_sample(
    img: np.ndarray,
    grid: np.ndarray,
    out: np.ndarray,
    mode: str = "bilinear",
) -> np.ndarray:
    """Numpy grid sampling algorithm

    params:
    - img (np.ndarray)
    - grid (np.ndarray)
    - out (np.ndarray)
    - mode (str): ('bilinear', 'bicubic', 'nearest')

    return:
    - out (np.ndarray)

    NOTE:
    - assumes that `img`, `grid`, and `out` have the same dimension of
      (batch, channel, height, width).
    - channel for `grid` should be 2 (yx)

    """

    if mode == "nearest":
        out = nearest(img, grid, out)
    elif mode == "bilinear":
        out = bilinear(img, grid, out)
    elif mode == "bicubic":
        out = bicubic(img, grid, out)
    else:
        raise ValueError(f"ERR: {mode} is not supported")

    return out

def grid_sample_override(
    img: np.ndarray,
    grid: np.ndarray,
    out: np.ndarray,
    mode: str = "bilinear",
) -> np.ndarray:
    """Numpy grid sampling algorithm

    params:
    - img (np.ndarray)
    - grid (np.ndarray)
    - out (np.ndarray)
    - mode (str): ('bilinear', 'bicubic', 'nearest')

    return:
    - out (np.ndarray)

    NOTE:
    - assumes that `img`, `grid`, and `out` have the same dimension of
      (batch, channel, height, width).
    - channel for `grid` should be 2 (yx)

    """
    grid_f = None
    if mode == "nearest":
        grid_f = nearest
    elif mode == "bilinear":
        grid_f = bilinear
    elif mode == "bicubic":
        grid_f = bicubic
    else:
        raise ValueError(f"ERR: {mode} is not supported")
    img = img[None, ...]
    for i, grd in enumerate(grid):
        grd = grd[None, ...]
        out_b = out[i]
        out_b = out_b[None, ...]
        out[i] = grid_f(img, grd, out_b).squeeze()

    return out
