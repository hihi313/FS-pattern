import argparse
from pathlib import Path
from typing import Optional

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import util as util


def imshow(
    img: np.ndarray,
    name: str = None,
    out_dir: str = None,
    hold: bool = False,
    ext: str = "jpg",
) -> Optional[plt.Figure]:
    if name is None:
        util.get_default_name(prefix="Plot @ ")

    if img is None or img.size == 0:
        raise ValueError("Empty or invalid image provided")

    # Handle different image formats
    if img.ndim == 2:
        cmap = "gray"
    elif img.shape[-1] == 1:
        cmap = "gray"
        img = img.squeeze()
    else:
        cmap = None
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Clear existing figure and create new one
    plt.close(name)
    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    ax.set_title(name)
    ax.axis("off")

    if out_dir is not None:
        ext = ext.lower()
        valid_exts = {"jpg", "png", "jpeg", "tiff", "bmp"}
        if ext not in valid_exts:
            raise ValueError(f"Invalid extension. Must be one of {valid_exts}")
        file_path = util.get_output_path(parent=out_dir, filename=name, ext=ext)
        fig.savefig(file_path)

    if not hold:
        plt.show()
        # Close all after show
        plt.close("all")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Display multiple images using OpenCV")
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="List of image file paths to display",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for image_path in args.input:
        if not image_path.exists() or not image_path.is_file():
            print(f"Skipping invalid file: {image_path}")
            continue

        image = cv.imread(str(image_path))
        imshow(image, hold=False)


if __name__ == "__main__":
    main()
