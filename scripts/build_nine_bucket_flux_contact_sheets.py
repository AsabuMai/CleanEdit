#!/usr/bin/env python3
"""Build source/result contact sheets for the locked nine-bucket FLUX audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def fit(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sheet-dir", type=Path, required=True)
    parser.add_argument("--bucket", action="append", help="Build only the named bucket; repeat as needed.")
    parser.add_argument("--tile", type=int, default=320)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    key_font = load_font(15)
    column_font = load_font(14)
    args.sheet_dir.mkdir(parents=True, exist_ok=True)
    for bucket, spec in lock["buckets"].items():
        if args.bucket and bucket not in args.bucket:
            continue
        cases = list(spec["cases"].items())
        case_width = args.tile * 2
        label_height = 58
        case_height = args.tile + label_height
        columns = 2
        rows = math.ceil(len(cases) / columns)
        sheet = Image.new("RGB", (case_width * columns, case_height * rows), (235, 235, 235))
        draw = ImageDraw.Draw(sheet)
        for index, (key, case) in enumerate(cases):
            x = (index % columns) * case_width
            y = (index // columns) * case_height
            source = fit(Image.open(case["source_image_path"]), args.tile)
            result_path = args.output_root / bucket / key / "dece_rf_flux" / "seed_10" / "result.png"
            result = fit(Image.open(result_path), args.tile)
            sheet.paste(source, (x, y + label_height))
            sheet.paste(result, (x + args.tile, y + label_height))
            draw.rectangle((x, y, x + case_width, y + label_height), fill=(20, 20, 20))
            draw.text((x + 8, y + 5), key, font=key_font, fill="white")
            draw.text((x + 8, y + 34), "source", font=column_font, fill=(180, 210, 255))
            draw.text(
                (x + args.tile + 8, y + 34),
                "H100 / no final postprocess",
                font=column_font,
                fill=(180, 255, 190),
            )
        out = args.sheet_dir / f"{bucket}.png"
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
