#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Remove tiny line artifacts in a narrow band outside a subject mask."
    )
    p.add_argument("--image", required=True)
    p.add_argument("--subject-mask", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--debug-mask")
    p.add_argument("--side", choices=["right", "left"], default="right")
    p.add_argument("--hp-threshold", type=int, default=2)
    p.add_argument("--inpaint-radius", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    mask_path = Path(args.subject_mask)
    out_path = Path(args.output)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    subject = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if subject is None:
        raise FileNotFoundError(mask_path)

    subject = cv2.resize(subject, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(subject > 128)
    if len(xs) == 0:
        raise ValueError("subject mask is empty")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0

    region = np.zeros(subject.shape, np.uint8)
    ry0 = max(0, y0 - int(0.02 * h))
    ry1 = min(image.shape[0], y0 + int(0.43 * h))
    if args.side == "right":
        rx0 = max(0, x1 - int(0.16 * w))
        rx1 = min(image.shape[1], x1 + int(0.42 * w))
    else:
        rx0 = max(0, x0 - int(0.42 * w))
        rx1 = min(image.shape[1], x0 + int(0.16 * w))
    region[ry0:ry1, rx0:rx1] = 255

    keep = cv2.dilate(
        (subject > 80).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 11)
    highpass = cv2.absdiff(gray, median)
    line_mask = ((highpass > args.hp_threshold) & (region > 0) & (keep == 0)).astype(np.uint8) * 255
    line_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    )
    line_mask = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    cleaned = cv2.inpaint(image, line_mask, args.inpaint_radius, cv2.INPAINT_TELEA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cleaned)
    if args.debug_mask:
        debug_path = Path(args.debug_mask)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), line_mask)
    print(f"saved {out_path} mask_pixels={(line_mask > 0).sum()}")


if __name__ == "__main__":
    main()
