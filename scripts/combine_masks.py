from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L").save(path)


def make_overlay(image_path: Path, mask: np.ndarray, output: Path) -> None:
    image = Image.open(image_path).convert("RGB").resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32)
    color = np.array([0.0, 160.0, 255.0], dtype=np.float32)
    alpha = np.clip(mask[..., None], 0.0, 1.0) * 0.55
    overlay = (rgb * (1.0 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine grayscale masks with max-union or intersection.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overlay-image", type=Path, default=None)
    parser.add_argument("--overlay-output", type=Path, default=None)
    parser.add_argument("--mode", choices=("max", "min", "multiply"), default="max")
    parser.add_argument("masks", nargs="+", type=Path)
    args = parser.parse_args()

    base = load_mask(args.masks[0])
    combined = base
    for path in args.masks[1:]:
        mask = load_mask(path, size=(base.shape[1], base.shape[0]))
        if args.mode == "max":
            combined = np.maximum(combined, mask)
        elif args.mode == "min":
            combined = np.minimum(combined, mask)
        else:
            combined = combined * mask

    save_mask(combined, args.output)
    if args.overlay_image is not None and args.overlay_output is not None:
        make_overlay(args.overlay_image, combined, args.overlay_output)


if __name__ == "__main__":
    main()
