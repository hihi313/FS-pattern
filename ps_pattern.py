# %% imports
import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# %% Setup/Init
matplotlib.use('Qt5Agg')
%matplotlib inline

# Path
ROOT = Path("/app")
IMG_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "outputs"

# %% Utils


def imshow(name: str, img: np.ndarray, path: Path = None):
    fig = plt.figure(name)
    if img.shape[-1] == 1 or img.ndim == 2:
        plt.imshow(img, cmap='gray')
        plt.colorbar()
    else:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    fig.show()
    if path is not None:
        cv.imwrite(str(path / f"{name}.png"), img, (cv.IMWRITE_PNG_COMPRESSION, 9,
                   cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_RLE))


def medialAxisTransform(img: np.ndarray, dist_type: int = cv.DIST_L2,
                        dist_maskSize: int = cv.DIST_MASK_PRECISE,
                        dilate_kernel: np.ndarray = None,
                        cmpop: int = cv.CMP_EQ,
                        extract: bool = True) -> np.ndarray:
    # Get MAT by distance transform
    distance = cv.distanceTransform(img, dist_type, dist_maskSize)
    # Structure elements (4-connectness local maximun, -1 is don't care)
    if dilate_kernel is None:
        dilate_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # Dilate to get local maximun
    # B&W dilation (that get max only)?
    dilated = cv.dilate(distance, dilate_kernel)
    # dilated = cv.morphologyEx(distance, cv.MORPH_DILATE, dilate_kernel) # seems the same as dilate()
    # If the pixel is local maximun, then it won't change its value
    mat = cv.compare(distance, dilated, cmpop)
    # Extract the skeleton/MAT
    if extract:
        mat = cv.bitwise_and(mat, img)
    return mat


# %% Process
img = cv.imread(str(IMG_DIR / "hand bw.png"), cv.IMREAD_REDUCED_GRAYSCALE_4) # IMREAD_GRAYSCALE IMREAD_REDUCED_GRAYSCALE_4
# To BW
_, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# Denoise
kernel = np.ones((15, 15), np.uint8)
# Delete noise dots
img_open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# Fill hole
img_close = cv.morphologyEx(img_open, cv.MORPH_CLOSE, kernel)

imshow('hand', img)
imshow('hand open', img_open)
imshow('hand close', img_close, OUTPUT_DIR)

# %% Fill the remaining white hole
# Get nonzero noise region
non_zero = np.nonzero(img_close[:20, :])
img_close[non_zero] = 0  # clear
imshow('hand close filled', img_close)

# %% Medial Axis Transform

img_bg = cv.bitwise_not(img_close)

img_mat = medialAxisTransform(img_close)
img_bg_mat = medialAxisTransform(img_bg, dist_type=cv.DIST_L1)

imshow("hand MAT", img_mat, OUTPUT_DIR)
imshow("hand BG MAT", img_bg_mat, OUTPUT_DIR)
# %% Test



# %%
# input()
