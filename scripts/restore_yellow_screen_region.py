from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_polygon(text: str, size: tuple[int, int]) -> list[tuple[float, float]]:
    width, height = size
    points: list[tuple[float, float]] = []
    for part in text.split(";"):
        if not part.strip():
            continue
        x_text, y_text = part.split(",")
        points.append((float(x_text.strip()) * width, float(y_text.strip()) * height))
    if len(points) < 3:
        raise ValueError("--polygon must contain at least three normalized x,y points")
    return points


def odd_filter_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore a yellow-screen object from the source using a color-derived mask."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--polygon", required=True, help="Normalized x,y polygon limiting the object search.")
    parser.add_argument("--dilate", type=int, default=19)
    parser.add_argument("--extra-dilate", type=int, default=9)
    parser.add_argument("--erode", type=int, default=5)
    parser.add_argument("--blur", type=float, default=0.5)
    parser.add_argument("--mask-output", type=Path, default=None)
    parser.add_argument("--overlay-output", type=Path, default=None)
    args = parser.parse_args()

    result = Image.open(args.result).convert("RGB")
    source = Image.open(args.source).convert("RGB").resize(result.size, Image.Resampling.LANCZOS)
    source_arr = np.asarray(source, dtype=np.float32) / 255.0
    r = source_arr[..., 0]
    g = source_arr[..., 1]
    b = source_arr[..., 2]

    poly_mask = Image.new("L", result.size, 0)
    ImageDraw.Draw(poly_mask).polygon(parse_polygon(args.polygon, result.size), fill=255)
    poly = np.asarray(poly_mask, dtype=np.uint8) > 0

    yellow = (r > 0.45) & (g > 0.42) & (b < 0.32) & ((r - b) > 0.25) & ((g - b) > 0.22)
    mask = Image.fromarray(((yellow & poly).astype(np.uint8) * 255), mode="L")
    mask = mask.filter(ImageFilter.MaxFilter(odd_filter_size(args.dilate)))
    mask = mask.filter(ImageFilter.MaxFilter(odd_filter_size(args.extra_dilate)))
    mask = mask.filter(ImageFilter.MinFilter(odd_filter_size(args.erode)))

    mask_arr = np.asarray(mask, dtype=np.uint8) > 0
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    saturation = (maxc - minc) / np.maximum(maxc, 1e-6)
    red = (r > g * 1.15) & (r > b * 1.15) & (saturation > 0.16) & (maxc > 0.12)
    mask_arr = mask_arr & poly & (~red)
    mask = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L")
    if args.blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(args.blur)))

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
