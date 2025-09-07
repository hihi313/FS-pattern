from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2 as cv
import img2pdf
import numpy as np
from loguru import logger
from PIL import Image


import util


def grayscale_denoise(
    img_bgr: np.ndarray,
    kern_size: int = 11,
):
    img_ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
    # Cr channel make skin color more stands out
    y, cr, cb = cv.split(img_ycrcb)

    # Median to remove noise
    img_mid = cv.medianBlur(cr, kern_size)

    # Morphological open & closing to remove noise & smooth local pixel value
    # kern_morph = cv.getStructuringElement(
    #     shape=cv.MORPH_ELLIPSE, ksize=(kern_size, kern_size)
    # )
    # img_open = cv.morphologyEx(img_mid, cv.MORPH_OPEN, kern_morph)
    # img_close = cv.morphologyEx(img_open, cv.MORPH_CLOSE, kern_morph)

    # util.imshow(cr, name="Cr channel")
    # util.imshow(img_mid, name="Cr channel after median blur")
    # util.imshow(img_open, name="Cr channel after morphological open")
    # util.imshow(img_close, name="Cr channel after morphological close")

    return img_mid


def parse_args():
    parser = argparse.ArgumentParser(description="Display multiple images using OpenCV")
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="List of image file paths to display",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs"),
        help="Path to save the output",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=300,
        help="DPI for converting inches to pixels (default: 300)",
    )
    parser.add_argument(
        "--paper_size",
        nargs=2,
        type=float,
        default=(8.3, 11.7),  # Default to A4 size in inches
        help="Paper size in inches (width height), default is A4 (8.3 x 11.7 inches)",
    )
    parser.add_argument(
        "--size-unit",
        type=str,
        choices=["in", "cm"],
        default="in",
        help="Unit for paper size, either 'in' or 'cm'. Default is 'in'.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=5,
        help="Dilation size in mm. Expand the hand region by this amount in mm",
    )
    parser.add_argument(
        "--output-size",
        nargs=2,
        type=float,
        default=(11.7, 16.5),  # Default to A3 size in inches
        help="Paper size in inches (width height), default is A4 (8.3 x 11.7 inches)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Set the logging level (default: DEBUG)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="pdf",
        help="Output file extension. Can be png. (default: pdf) "
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set log level
    logger.remove()
    logger.add(sys.stderr, level=args.log_level) 

    # All convert to inches, since DPI is in inches
    if args.size_unit == "cm":
        args.paper_size = util.mm_to_in(np.array(args.paper_size) * 10)
        args.output_size = util.mm_to_in(np.array(args.output_size) * 10)


    for image_path in args.input:
        if not image_path.exists() or not image_path.is_file():
            print(f"Skipping invalid file: {image_path}")
            continue

        img = cv.imread(str(image_path))
        img = util.resize(img, (1000,))
        H, W = img.shape[:2]
        print(f"Image shape: {H}x{W}")

        gray = grayscale_denoise(img)
        cv.imshow("Grayscale Denoise", gray)

        th, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        cv.imshow("Binary Threshold", bw)

        contours, hierarchy = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        cntr_info = []
        for i, paper_cntr in enumerate(contours):
            paper_hull = cv.convexHull(paper_cntr)
            hull_area = cv.contourArea(paper_hull)
            hull_area_rate = hull_area / (H * W)
            # TODO: +args to set hull_area_rate threshold
            if hull_area_rate < 0.5 or hull_area_rate >= 0.98:
                logger.debug(f"Contour {i} skipped: Hull area rate = {hull_area_rate}")
                continue
            else:
                cntr_info.append(
                    {
                        "contour": paper_cntr,
                        "hull": paper_hull,
                        "hull_area": hull_area,
                        "hull_area_rate": hull_area_rate,
                    }
                )
                logger.info(
                    f"Contour {i}: Hull area = {hull_area}, Hull area rate = {hull_area_rate}"
                )

        cntr_info.sort(key=lambda x: x["hull_area"], reverse=True)

        # Take the largest contour as the paper contour
        paper_cntr_info = cntr_info[0]

        paper_cntr = paper_cntr_info["contour"]
        paper_hull = paper_cntr_info["hull"]

        # Find approximate quadrilateral
        paper_poly4 = cv.approxPolyN(paper_hull, 4, ensure_convex=False)
        paper_poly4 = util.order_points(paper_poly4)

        img_cntr_poly = np.zeros((H, W, 3), dtype=np.uint8)
        cv.drawContours(img_cntr_poly, [paper_cntr], -1, color=(255, 0, 0), thickness=1)
        cv.drawContours(
            img_cntr_poly, [paper_poly4], -1, color=(0, 255, 0), thickness=1
        )
        cv.imshow(f"Paper's poly", img_cntr_poly)

        # straighten paper size
        dst_size = util.inche_to_px(args.paper_size, args.dpi)  # (W,H)
        # straighten paper corners
        dst_corner = np.array(
            [
                [0, 0],
                [dst_size[0], 0],
                [dst_size[0], dst_size[1]],
                [0, dst_size[1]],
            ],
            dtype=np.float64,
        )
        # solve the homography matrix by SVD (most robust)
        homo_mat = cv.getPerspectiveTransform(
            paper_poly4.squeeze().astype(np.float32), 
            dst_corner.astype(np.float32), 
            solveMethod=cv.DECOMP_SVD
        )

        # Close the hand region to ensure it form a connected component
        paper_poly4 = paper_poly4.squeeze()
        # TODO: change thickness to ?% width
        bw = cv.line(bw, paper_poly4[-1], paper_poly4[-2], (0, 0, 0), 2)
        cv.imshow("BW hand region", bw)

        # Find hand connected component
        numLabels, labels, stats, centroids = cv.connectedComponentsWithStatsWithAlgorithm(bw, 8,cv.CV_32S,cv.CCL_BOLELLI)

        # Sort components from large area
        comp_areas = sorted(
            [(lbl, stats[lbl, cv.CC_STAT_AREA]) for lbl in range(1, numLabels)],
            reverse=True,
            key=lambda x: x[1],
        )

        # Find the component where its centroid inside the paper
        hand_lbl = None
        for lbl, area in comp_areas:
            cx, cy = centroids[lbl]
            if cv.pointPolygonTest(paper_poly4, (cx, cy), False) > 0: 
                hand_lbl = lbl
                break
        hand = (labels == hand_lbl).astype(np.uint8) * 255
        cv.imshow('hand', hand)

        # Unwrap
        hand_unwrap = cv.warpPerspective(
            hand,
            homo_mat,
            dst_size,
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
        )
        cv.imshow("Unwrap hand", hand_unwrap)

        # TODO: close & find connected componet after wrap
        # TODO: handle gray scale problem, the image is actually gray scale, not B/W after some operations
        # TODO: wrap paper extract & unwrap to class

        # Move to output/larger paper
        Wo, Ho = util.inche_to_px(args.output_size, args.dpi)  # (W,H)
        hand_pad = util.pad_to_size(hand_unwrap, Ho, Wo, 0)
        cv.imshow("Hand pad", hand_pad)

        # Dilate r mm
        # Convert r from mm to px
        r = util.mm_to_px(args.radius, dpi=args.dpi)
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (2*r + 1, 2*r + 1)
        )  # creates an 11×11 ellipse mask

        hand_cntrs, _ = cv.findContours(hand_pad, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
        hand_cntr = hand_cntrs[0]

        # Dilation
        hand_dilate = cv.dilate(hand_pad, kernel, iterations=1, borderType=cv.BORDER_REPLICATE)
        cv.imshow("Hand dilated", hand_dilate)
        cv.imwrite("Hand dilated.png", hand_dilate)

        # Convert to black contour only (on white background). The output pdf file dimension verified OK
        # TODO: + args to set the contour thickness
        hand_dilate_cntrs, _ = cv.findContours(hand_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        hand_cntr_img = np.ones_like(hand_dilate) * 255
        cv.drawContours(
            hand_cntr_img, hand_dilate_cntrs, -1, (0, 0, 0), thickness=1
        )

        # Draw reference line w/ specified length
        hand_dilate_cntr = hand_dilate_cntrs[0]
        x,y,w,h = cv.boundingRect(hand_dilate_cntr)
        ref_len = util.mm_to_px(100, args.dpi) # 10 cm
        margin = util.mm_to_px(5, args.dpi)
        # Vertical line at top left
        cv.line(hand_cntr_img, (x-margin, y ), (x-margin, y+ref_len), (0, 0, 0), thickness=3)
        # Horizontal line at top
        cv.line(hand_cntr_img, (x, y-margin), (x+ref_len, y-margin), (0, 0, 0), thickness=3)
        cv.drawContours(
            hand_cntr_img, [hand_cntr], -1, (0, 0, 0), thickness=1
        )
        cv.imshow("Hand dilate contour", hand_cntr_img)

        # Save as file with DPI
        out_path = util.get_output_path(args.output, prefix=f"{image_path.stem}_", ext=args.ext)
        img_pil = Image.fromarray(hand_cntr_img)
        if args.ext.lower() in "pdf":
            # Save the image to an in-memory buffer
            byte_buffer = io.BytesIO()
            img_pil.save(byte_buffer, format='PNG')
            # Get the bytes from the buffer
            image_bytes = byte_buffer.getvalue()
            # Save bytes img to pdf
            layout_fun = img2pdf.get_fixed_dpi_layout_fun((args.dpi, args.dpi))
            with open(out_path, "wb") as f:
                f.write(img2pdf.convert(image_bytes, layout_fun=layout_fun))
        else:
            img_pil.save(out_path, dpi=(args.dpi, args.dpi))

        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
