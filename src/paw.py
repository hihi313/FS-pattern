# %% imports
import cv2 as cv
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from typing import List
import sklearn as sk
import pandas as pd
from sklearn.cluster import DBSCAN

# %% Setup/Init
# matplotlib.use('Qt5Agg')
# %matplotlib inline

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



# %% Get finger points
idx = np.argwhere(img_mat > 0) # Get MAT pixels' indices
clusters = DBSCAN(eps=15, min_samples=3).fit(idx)
skeleton_lbl = pd.DataFrame(data={
    "skeleton_x": idx[:, 1], # column=x
    "skeleton_y": idx[:, 0], # row=y
    "label": clusters.labels_
})
# skeleton_lbl = skeleton_lbl[(skeleton_lbl["label"] >= 0) & (skeleton_lbl["label"] <= 4)]

plt.scatter(skeleton_lbl["skeleton_x"], skeleton_lbl["skeleton_y"], c=skeleton_lbl["label"], cmap='hsv')
plt.colorbar()
plt.show()

# %% Show
# (x,y) = (x0,y0) + s*(vx,vy)
class Line:
    def __init__(self, vx: float, vy: float, x0: float, y0: float) -> None:
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.vx = float(vx)
        self.vy = float(vy)
        self.horizontal = (self.vy == 0)
        self.vertical = (self.vx == 0)

    def y(self, x: float)->float:
        if self.vertical:
            return None
        else:
            return self.y0 + (float(x) - self.x0) * self.vy / self.vx
    
    def x(self, y: float)->float:
        if self.horizontal:
            return None
        else:
            return self.x0 + (float(y) - self.y0) * self.vx / self.vy
        
    # Input: x, y range
    def get_endpoints(self, x: List[float], y: List[float]):
        x1 = x2 = self.x0
        y1 = y2 = self.y0
        lx = len(x) if x is not None else 0
        ly = len(y) if y is not None else 0
        if self.horizontal and lx > 0: 
            # TODO: horizontal & x = None?
            x1, y1 = min(x), self.y0
            x2, y2 = max(x), self.y0
        elif self.vertical and ly > 0:
            x1, y1 = self.x0, min(y)
            x2, y2 = self.x0, max(y)
        else:
            useX = True
            if y is None or ly <= 0:
                # x only
                useX = True
            elif x is None or lx <= 0:
                # y only
                useX = False
            else:
                # x & y
                stdx = np.std(x)
                stdy = np.std(y)
                if stdx >= stdy:
                    # Use x
                    useX = True
                else:
                    # Use y
                    useX = False
            if useX:
                x.sort()
                x1, y1 = x[0], self.y(x[0])
                x2, y2 = x[-1], self.y(x[-1])
            else:
                y.sort()
                x1, y1 = self.x(y[0]), y[0]
                x2, y2 = self.x(y[-1]), y[-1]
        return (x1, y1), (x2, y2)

# for i in [0, 1, 2, 3, 4]:
#     finger_i = fingers[fingers["label"] == i].to_numpy()[:, 0:2]
#     line = cv.fitLine(finger_i, cv.DIST_L2, 0, 0.01, 0.01)
#     line_func = Line(line[0], line[1], line[2], line[3])
#     ymax = max(finger_i[:, 1])
#     ymin = min(finger_i[:, 1])
#     tmp2 = cv.line(tmp, (line_func.x(ymax), ymax), (line_func.x(ymin), ymin), (255, 255, 255), 1, cv.LINE_8, 0)
# imshow("tmp2", tmp2, OUTPUT_DIR)

# %% MAT skeleton line fitting

# BW image
def regress_lines(img: np.ndarray)-> List[Line]:
    # Get pixels' index
    idx = np.argwhere(img > 0) 
    # Group & exlude outlier
    clusters = DBSCAN(eps=15, min_samples=3).fit(idx)
    skeletons = pd.DataFrame(data={
        "x": idx[:, 1], # column=x
        "y": idx[:, 0], # row=y
        "group": clusters.labels_
    })
    plt.scatter(skeletons["x"], skeletons["y"], c=skeletons["group"], cmap='hsv')
    plt.colorbar()
    plt.show()

    lines = []
    for grp in set(clusters.labels_):
        # -1 = outlier group
        if grp >= 0:
            # Get all (x, y) in the group
            skel = skeletons[skeletons["group"] == grp].to_numpy()[:, 0:2]
            line_params = cv.fitLine(skel, cv.DIST_L2, 0, 0.01, 0.01)
            lines.append({
                "x": skel[:, 0],
                "y": skel[:, 1],
                "f": Line(line_params[0], line_params[1], line_params[2], line_params[3])
            })
    return lines

segments = regress_lines(img_bg_mat)

tmp = img_bg_mat.copy()

for seg in segments:
    p1, p2 = seg["f"].get_endpoints(None, seg["y"])
    p1, p2 =(round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1]))
    tmp2 = cv.line(tmp, p1, p2, (255, 255, 255), 1, cv.LINE_8, 0)

imshow("tmp2", tmp2, OUTPUT_DIR)

# %%
# input()