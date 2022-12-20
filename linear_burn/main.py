#!/usr/bin/env python

from PIL import Image
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fg = io.imread("foreground.png")
    color = io.imread("color.png")
    gt = io.imread("gt.png")

    hat = fg[..., :3] / 255 + color[..., :3] / 255 - 1
    hat = (np.clip(hat, 0, 1) * 255).astype('uint8')
    hat = np.concatenate([hat, fg[..., -1:]], axis=-1)
    # Figure
    plt.rcParams["figure.figsize"] = (30, 12)

    plt.subplot(1, 4, 1)
    plt.title("Foreground", fontsize=20)
    plt.imshow(fg)

    plt.subplot(1, 4, 2)
    plt.title("Base Color", fontsize=20)
    plt.imshow(color)

    plt.subplot(1, 4, 3)
    plt.title("Reproduce", fontsize=20)
    plt.imshow(hat)

    plt.subplot(1, 4, 4)
    plt.title("GT", fontsize=20)
    plt.imshow(gt)

    plt.tight_layout()
    plt.savefig("plot.jpg", dpi=200)
