from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_polygon(text: str, size: tuple[int, int]) -> list[tuple[int, int]]:
    width, height = size
    points: list[tuple[int, int]] = []
    for part in text.split(";"):
        if not part.strip():
            continue
        x_text, y_text = part.split(",")
        x = float(x_text.strip())
        y = float(y_text.strip())
        points.append((int(round(x * width)), int(round(y * height))))
    if len(points) < 3:
        raise ValueError("--polygon must contain at least three x,y points separated by semicolons")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a polygonal region from the source image into a result.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--polygon", required=True)
    parser.add_argument("--feather", type=float, default=1.5)
    parser.add_argument("--exclude-source-red", action="store_true", default=False)
    parser.add_argument("--mask-output", type=Path, default=None)
    parser.add_argument("--overlay-output", type=Path, default=None)
    args = parser.parse_args()

    result = Image.open(args.result).convert("RGB")
    source = Image.open(args.source).convert("RGB").resize(result.size, Image.Resampling.LANCZOS)
    mask = Image.new("L", result.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(parse_polygon(args.polygon, result.size), fill=255)
    if args.feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(args.feather)))
    if args.exclude_source_red:
        source_arr = np.asarray(source, dtype=np.float32) / 255.0
        r = source_arr[..., 0]
        g = source_arr[..., 1]
        b = source_arr[..., 2]
        maxc = np.maximum.reduce([r, g, b])
        minc = np.minimum.reduce([r, g, b])
        saturation = (maxc - minc) / np.maximum(maxc, 1e-6)
        red = (r > g * 1.18) & (r > b * 1.18) & (saturation > 0.18) & (maxc > 0.12)
        mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
        mask_arr = np.where(red, 0.0, mask_arr)
        mask = Image.fromarray((mask_arr * 255.0).round().astype(np.uint8), mode="L")

    output = Image.composite(source, result, mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    if args.mask_output is not None:
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.mask_output)
    if args.overlay_output is not None:
        result_arr = np.asarray(result, dtype=np.float32)
        color = np.array([255.0, 180.0, 0.0], dtype=np.float32)
        alpha = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None] * 0.45
        overlay = (result_arr * (1.0 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(args.overlay_output)


if __name__ == "__main__":
    main()
