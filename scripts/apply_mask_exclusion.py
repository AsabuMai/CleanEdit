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
        points.append((int(round(float(x_text) * width)), int(round(float(y_text) * height))))
    if len(points) < 3:
        raise ValueError("--polygon must contain at least three x,y points separated by semicolons")
    return points


def make_overlay(image_path: Path, mask: np.ndarray, output: Path) -> None:
    image = Image.open(image_path).convert("RGB").resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32)
    color = np.array([0.0, 160.0, 255.0], dtype=np.float32)
    alpha = np.clip(mask[..., None], 0.0, 1.0) * 0.55
    overlay = (rgb * (1.0 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtract a polygonal exclusion region from a grayscale mask.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--polygon", required=True)
    parser.add_argument("--feather", type=float, default=0.0)
    parser.add_argument("--overlay-image", type=Path, default=None)
    parser.add_argument("--overlay-output", type=Path, default=None)
    parser.add_argument("--exclusion-output", type=Path, default=None)
    args = parser.parse_args()

    mask_image = Image.open(args.input).convert("L")
    exclusion = Image.new("L", mask_image.size, 0)
    ImageDraw.Draw(exclusion).polygon(parse_polygon(args.polygon, mask_image.size), fill=255)
    if args.feather > 0:
        exclusion = exclusion.filter(ImageFilter.GaussianBlur(float(args.feather)))

    mask = np.asarray(mask_image, dtype=np.float32) / 255.0
    exclude = np.asarray(exclusion, dtype=np.float32) / 255.0
    output = np.clip(mask * (1.0 - exclude), 0.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((output * 255.0).round().astype(np.uint8), mode="L").save(args.output)
    if args.exclusion_output is not None:
        args.exclusion_output.parent.mkdir(parents=True, exist_ok=True)
        exclusion.save(args.exclusion_output)
    if args.overlay_image is not None and args.overlay_output is not None:
        make_overlay(args.overlay_image, output, args.overlay_output)


if __name__ == "__main__":
    main()
