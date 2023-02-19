# %% imports
import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# %% Setup/Init
matplotlib.use('Qt5Agg')
# %matplotlib inline

# path
ROOT = Path("/app")
IMG_DIR = ROOT / "images"

# %% Utils


def imshow(name: str, img: np.ndarray, save: bool=False):
    fig = plt.figure(name)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    fig.show()
    if save:
        path = IMG_DIR / f"{name}.png"
        cv.imwrite(str(path), img, (cv.IMWRITE_PNG_COMPRESSION, 9,
                   cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_RLE))


# %% Process

img = cv.imread(str(IMG_DIR / "hand bw.png"), cv.IMREAD_GRAYSCALE)
_, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

kernel = np.ones((15, 15), np.uint8)
img_open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
img_close = cv.morphologyEx(img_open, cv.MORPH_CLOSE, kernel)

imshow('hand', img)
imshow('hand open', img_open, save=True)
imshow('hand close', img_close, save=True)

# %% Fill the remaining white hole
non_zero = np.nonzero(img_close[:20, :])
# print(non_zero.ama)
img_close[non_zero] = 0
imshow('hand close filled', img_close, save=True)

# %% Medial Axis Transform
img_dist = cv.distanceTransform(img_close, cv.DIST_L2, cv.DIST_MASK_PRECISE)
img_dist = cv.normalize(img_dist, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

imshow("img_dist", img_dist)

# %%

se_sqr = np.ones((3, 3), np.uint8)
se_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
se = np.array((
        [-1, 0, -1],
        [0, 0, 0],
        [-1, 0, -1]), dtype="int")
img_dist_dilate = cv.dilate(img_dist, se_sqr)

# Find the pixels that are equal to the dilated image
maxima = cv.compare(img_dist, img_dist_dilate, cv.CMP_EQ)

imshow("img_dist_dilate", img_dist_dilate)
imshow("maxima", maxima, save=True)

non_plateau = cv.erode(img_dist, se_sqr)
non_plateau2 = cv.compare(img_dist, non_plateau, cv.CMP_GT)
out = cv.bitwise_and(maxima, non_plateau2)

imshow("out", out)
# %%
input()
