from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
OUT = ROOT / "paper" / "assets" / "appendix_grids" / "samflow_seed10_phase2_preview.png"

TASKS = [
    ("cat_crown", "T1 cat crown"),
    ("dog_bow_tie_phase2", "T1 dog bow tie"),
    ("dog_front_sunglasses_phase2", "T1 dog sunglasses"),
    ("bowl_apple_inside", "T2 bowl apple"),
    ("white_bowl_orange_tabletop_phase2", "T2 white bowl orange"),
    ("brown_bowl_lemon_phase2", "T2 brown bowl lemon"),
    ("tshirt_star", "T3 t-shirt star"),
    ("mug_heart", "T3 mug heart"),
    ("tote_leaf", "T3 tote leaf"),
    ("red_office_chair_to_blue_office_chair", "T4 chair recolor"),
    ("green_mug_orange_phase2", "T4 mug recolor"),
    ("yellow_vase_blue_phase2", "T4 vase recolor"),
    ("pillow_same_color_cable_knit", "T5 white pillow"),
    ("pillow_same_color_cable_knit_grey", "T5 grey pillow"),
    ("pillow_same_color_cable_knit_armchair", "T5 armchair pillow"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for path in [
        EXP / "e2_t1_t4_formal_baseline_manifest.csv",
        EXP / "e2_t5_formal_baseline_manifest.csv",
    ]:
        for row in read_rows(path):
            lookup.setdefault(row.get("task", ""), row.get("source_image", ""))
    return lookup


def fit(path: Path | None, size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), "white")
    if not path or not path.exists():
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, size - 1, size - 1), outline=(180, 180, 180))
        draw.text((12, size // 2 - 8), "missing", fill=(130, 0, 0))
        return canvas
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def main() -> int:
    sources = source_lookup()
    columns = [
        ("source", "Source"),
        ("sam_flow_sd3", "Sam-Flow SD3"),
        ("sam_flow_flux", "Sam-Flow FLUX"),
    ]
    thumb = 168
    label_w = 230
    header_h = 42
    pad = 12
    width = label_w + len(columns) * (thumb + pad) + pad
    height = header_h + len(TASKS) * (thumb + pad) + pad
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    for col_idx, (_, label) in enumerate(columns):
        x = label_w + col_idx * (thumb + pad)
        draw.text((x, 14), label, fill=(0, 0, 0), font=font)

    y = header_h
    for task, label in TASKS:
        draw.text((pad, y + thumb // 2 - 8), label, fill=(0, 0, 0), font=font)
        for col_idx, (method, _) in enumerate(columns):
            x = label_w + col_idx * (thumb + pad)
            if method == "source":
                rel = sources.get(task, "")
                path = ROOT / rel if rel else None
            else:
                path = ROOT / "outputs" / "baselines" / method / task / "seed_10" / "result.png"
            out.paste(fit(path, thumb), (x, y))
        y += thumb + pad

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
