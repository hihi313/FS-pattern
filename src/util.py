from __future__ import annotations

from datetime import datetime
from pathlib import Path
import struct
from typing import Iterable, Optional, Sequence, Tuple, Union
import zlib
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def get_default_name(prefix: str = None, time_fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate a default name based on the current datetime.

    Args:
        time_fmt (str): The format for the datetime string.

    Returns:
        str: A formatted string representing the current datetime.
    """
    return f"{prefix}{datetime.now().strftime(time_fmt)}"


def get_output_path(parent: str = "./", filename: str = None, ext: str = "jpg") -> str:
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = get_default_name(prefix="output_")
    return f"{parent / f'{filename}.{ext}'}"


def imshow(
    img: np.ndarray,
    name: str = None,
    hold: bool = True,
    out_dir: str = None,
) -> Optional[plt.Figure]:
    if name is None:
        get_default_name(prefix="Plot @ ")

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
    fig, ax = plt.subplots(num=name)
    ax.imshow(img, cmap=cmap)
    ax.set_title(name)
    ax.axis("off")

    if out_dir is not None:
        imsave(img, path=out_dir, filename=name)

    plt.show(block=not hold)
    return fig


def imsave(
    img: np.ndarray,
    path: str,
    filename: str = None,
    ext: str = "png",
    convert: int = None,
) -> None:
    if convert is not None:
        img = cv.cvtColor(img, convert)

    path = get_output_path(parent=path, filename=filename, ext=ext)

    if filename is None:
        filename = get_default_name(prefix="Plot_")

    valid_exts = {"jpg", "png", "jpeg", "tiff", "bmp"}
    if ext.lower() not in valid_exts:
        raise ValueError(f"Invalid extension. Must be one of {valid_exts}")

    cv.imwrite(path, img)


import cv2
import numpy as np


def resize(
    img: np.ndarray, target_size=None, scale=None, interpolation=cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize an image either by specifying a target size or a scale factor.

    Args:
        img (np.ndarray): Input image.
        target_size:
            - int              : square output of (int, int)
            - tuple/list of 1 : (H,) → compute W to preserve aspect ratio
            - tuple/list of 2 : (H, W) where H or W may be None → compute missing dim
        scale:
            - float            : uniform scale for both dims
            - tuple/list of 1 : (s_h,) → only scale height, width unchanged
            - tuple/list of 2 : (s_h, s_w) where s_h or s_w may be None → 1.0 for missing
        interpolation: one of the cv2.INTER_* flags

    Returns:
        np.ndarray: The resized image.

    Raises:
        ValueError: if neither or both of target_size/scale are provided, or invalid specs.
        TypeError:  if passed wrong types.
    """
    if (target_size is None) == (scale is None):
        raise ValueError("Specify exactly one of target_size or scale.")

    h, w = img.shape[:2]

    # ----------------------------
    # 1) Resize by absolute target
    # ----------------------------
    if target_size is not None:
        # scalar → square
        if isinstance(target_size, (int, float)):
            new_h = new_w = int(target_size)
        # sequence
        elif isinstance(target_size, (tuple, list)):
            if len(target_size) == 1:
                # (H,) → preserve aspect ratio
                new_h = int(target_size[0])
                ratio = new_h / h
                new_w = int(w * ratio)
            elif len(target_size) == 2:
                t_h, t_w = target_size
                # both None → invalid
                if t_h is None and t_w is None:
                    raise ValueError("At least one of H or W must be non‐None.")
                # compute missing dimension
                if t_h is None:
                    new_w = int(t_w)
                    ratio = new_w / w
                    new_h = int(h * ratio)
                elif t_w is None:
                    new_h = int(t_h)
                    ratio = new_h / h
                    new_w = int(w * ratio)
                else:
                    new_h, new_w = int(t_h), int(t_w)
            else:
                raise ValueError("target_size must be int, 1‐tuple, or 2‐tuple.")
        else:
            raise TypeError("target_size must be int, float, tuple, or list.")

        return cv2.resize(img, dsize=(new_w, new_h), interpolation=interpolation)

    # ----------------------------
    # 2) Resize by scale factors
    # ----------------------------
    else:
        # scalar → uniform scale
        if isinstance(scale, (int, float)):
            fx = fy = float(scale)
        # sequence
        elif isinstance(scale, (tuple, list)):
            if len(scale) == 1:
                # (s_h,) → height only
                fy = float(scale[0])
                fx = 1.0
            elif len(scale) == 2:
                s_h, s_w = scale
                # both None → invalid
                if s_h is None and s_w is None:
                    raise ValueError("At least one of s_h or s_w must be non‐None.")
                fy = float(s_h) if s_h is not None else 1.0
                fx = float(s_w) if s_w is not None else 1.0
            else:
                raise ValueError("scale must be float, 1‐tuple, or 2‐tuple.")
        else:
            raise TypeError("scale must be int, float, tuple, or list.")

        return cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation=interpolation)


def order_points(pts: np.ndarray) -> np.ndarray:
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts, dtype=np.float64)

    old_shape = pts.shape
    pts = pts.squeeze()

    # Validate input shape and type.
    if pts.shape != (4, 2):
        raise ValueError(
            "Input points must be a NumPy array with shape (4, 2) after squeeze."
        )
    
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Return the ordered points.
    return np.array([
        pts[np.argmin(s)],  # Top-left: smallest sum
        pts[np.argmin(diff)],  # Top-right: smallest difference (x - y)
        pts[np.argmax(s)],  # Bottom-right: largest sum
        pts[np.argmax(diff)],  # Bottom-left: largest difference
    ], dtype=pts.dtype).reshape(old_shape)

def inche_to_px(inch: float | np.ndarray, dpi: float = 300) -> int:
    n_px = inch * dpi
    if isinstance(n_px, np.ndarray):
        return n_px.round().astype(int)
    return round(n_px)

def mm_to_px(mm: float | np.ndarray, dpi: float = 300) -> int:
    return inche_to_px(mm_to_in(mm), dpi=dpi)   


def mm_to_in(mm: float | np.ndarray | Iterable) -> float | np.ndarray | Iterable:
    if isinstance(mm, Iterable) and not isinstance(mm, np.ndarray):
        return [mm_to_in(m) for m in mm]
    else:
        return mm / 25.4
    

def imwrite_png(im: np.ndarray, filename: str, dpi: tuple = (300, 300)):
    """
    Saves an image as PNG with embedded DPI by finding the IDAT chunk.

    This version inserts the pHYs chunk immediately before the first IDAT
    chunk, making it robust for PNGs with variable-length headers.
    """
    # --- 1. Encode image to a PNG byte buffer ---
    retval, png_buffer = cv2.imencode(".png", im)
    if not retval:
        raise IOError("Failed to encode image to PNG format.")
    
    png_bytes = png_buffer.tobytes()

    # --- 2. Create the pHYs chunk ---
    # The pHYs chunk contains pixels per unit for X and Y axes.
    # The PNG specification requires units in meters.
    METERS_PER_INCH = 0.0254
    pixels_per_meter_x = int(round(dpi[0] / METERS_PER_INCH))
    pixels_per_meter_y = int(round(dpi[1] / METERS_PER_INCH))
    unit_specifier = b"\x01" # 1 = meters

    # Pack the chunk data (9 bytes total).
    # Format: Big-endian, 2 Unsigned Ints, 1 char byte.
    phys_chunk_data = struct.pack("!IIc", pixels_per_meter_x, pixels_per_meter_y, unit_specifier)
    
    # A complete PNG chunk includes its length, type, data, and CRC.
    phys_chunk = (
        struct.pack("!I", 9) +                # Length (4 bytes)
        b"pHYs" +                             # Chunk Type (4 bytes)
        phys_chunk_data +                     # Chunk Data (9 bytes)
        struct.pack("!I", zlib.crc32(b"pHYs" + phys_chunk_data)) # CRC (4 bytes)
    )

    # --- 3. Find the first IDAT chunk and insert pHYs before it ---
    # Every PNG with image data has at least one IDAT chunk.
    # The chunk structure is [Length][Type][Data][CRC]. We find the 'IDAT'
    # type and go back 4 bytes to the start of its Length field.
    try:
        idat_offset = png_bytes.index(b'IDAT') - 4
    except ValueError:
        raise IOError("Invalid PNG format: IDAT chunk not found.")

    final_png_bytes = (
        png_bytes[:idat_offset] +
        phys_chunk +
        png_bytes[idat_offset:]
    )

    # --- 4. Write the modified bytes to a file ---
    with open(filename, "wb") as out_file:
        out_file.write(final_png_bytes)

def rgb_to_gray_cv(rgb: Sequence[int]) -> int:
    """
    Converts an RGB color tuple to a grayscale scalar value using OpenCV's
    cv2.cvtColor() function.
    """
    if len(rgb) < 3:
        raise ValueError("Input must have at least 3 elements (R, G, B).")

    # Create a 1x1x3 pixel image for cvtColor
    pximg_rgb = np.array([[rgb[:3]]], dtype=np.uint8)
    return cv2.cvtColor(pximg_rgb, cv2.COLOR_RGB2GRAY).item()

def pad_to_size(
    img: np.ndarray,
    M: int,
    N: int,
    bg_color: Optional[Union[Tuple[int, int, int], int]] = None
) -> np.ndarray:
    """
    Place image at the center of a larger image with specified dimensions (M, N).
    """
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:  # Grayscale image
        H, W = img.shape
        C = 1

    # Check if the smaller image fits within the new dimensions
    if H > M or W > N:
        raise ValueError(f"The larger image ({M}, {N}) must >= the smaller image ({H}, {W}).")
    elif H == M and W == N:
        return img.copy()

    # Determine the background color based on the number of channels
    bg_color = img[0, 0].copy() if bg_color is None else bg_color

    # Create the new, blank M x N image with the background color
    if C == 1:
        large_img = np.full((M, N), bg_color, dtype=np.uint8)
    else: # Color image
        large_img = np.full((M, N, C), bg_color, dtype=np.uint8)

    # Calculate the coordinates to place the small image at the center
    y_start = (M - H) // 2
    x_start = (N - W) // 2
    y_end = y_start + H
    x_end = x_start + W

    # Place the small image onto the larger image using slicing
    large_img[y_start:y_end, x_start:x_end] = img

    return large_img