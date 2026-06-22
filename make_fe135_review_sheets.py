from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
EXP = PROJ / "experiments/flowedit135_fixedmask_metrics_20260621"
METRICS = EXP / "metrics.csv"
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest.json"
MASK_DIR = PROJ / "data/flowedit_compatible_135/eval_masks"
RUNS = PROJ / "outputs/flowedit135_metric_runs"

METHODS = [
    "source",
    "mask",
    "ours_sd3",
    "sam_flow_sd3",
    "flowedit_sd3",
    "splitflow_sd3",
    "sam_flow_flux",
    "otrf_enh_sd3",
    "fireflow",
    "reflex",
    "drfs_sd3",
]

SELECTED = [
    "fe_046_cat_crown_1_black_top_hat",
    "fe_094_dog_6_red_top_hat",
    "fe_208_pizza_tomato_olive_2_mushrooms",
    "fe_024_bus_2_volkswagen_logo",
    "fe_118_gas_station_1_cvpr",
    "fe_000_bear_1_black_bear",
    "fe_221_rocks_6_colorful_wooden_blocks",
    "fe_171_meditation_1_wooden_statue",
]


def thumb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    image = image.convert("RGB")
    image.thumbnail((size[0], size[1] - 20))
    canvas.paste(image, ((size[0] - image.width) // 2, 0))
    return canvas


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text[:34], fill=(0, 0, 0))


def source_image(item: dict) -> Image.Image:
    return Image.open(item["image"]).convert("RGB")


def mask_overlay(item: dict) -> Image.Image:
    base = source_image(item).convert("RGBA")
    mask = Image.open(MASK_DIR / f"{item['key']}_eval_mask.png").convert("L").resize(base.size)
    red = Image.new("RGBA", base.size, (255, 80, 30, 0))
    red.putalpha(mask.point(lambda v: int(v * 0.45)))
    return Image.alpha_composite(base, red).convert("RGB")


def method_image(key: str, method: str, item: dict) -> Image.Image:
    if method == "source":
        return source_image(item)
    if method == "mask":
        return mask_overlay(item)
    return Image.open(RUNS / key / method / "seed_10/result.png").convert("RGB")


def make_sheet(keys: list[str], name: str, method_names: list[str] = METHODS) -> None:
    manifest = {item["key"]: item for item in json.load(MANIFEST.open(encoding="utf-8"))}
    cell = (150, 145)
    left = 230
    header = 34
    width = left + cell[0] * len(method_names)
    height = header + cell[1] * len(keys)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col, method in enumerate(method_names):
        label(draw, (left + col * cell[0] + 4, 8), method)
    for row, key in enumerate(keys):
        item = manifest[key]
        y = header + row * cell[1]
        label(draw, (4, y + 8), key)
        label(draw, (4, y + 25), item.get("family_label", ""))
        label(draw, (4, y + 42), item.get("target_prompt", ""))
        for col, method in enumerate(method_names):
            try:
                img = method_image(key, method, item)
            except Exception:
                img = Image.new("RGB", (128, 128), (240, 240, 240))
            panel = thumb(img, cell)
            x = left + col * cell[0]
            sheet.paste(panel, (x, y))
    out = EXP / name
    sheet.save(out, quality=92)
    print(out)


def hard_keys() -> list[str]:
    rows = list(csv.DictReader(METRICS.open(newline="", encoding="utf-8")))
    by_task: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], {})[row["method"]] = row
    scored = []
    for task, group in by_task.items():
        ours = group.get("ours_sd3")
        sam = group.get("sam_flow_sd3")
        if not ours or not sam:
            continue
        gap = float(sam.get("local_clip_t") or 0) - float(ours.get("local_clip_t") or 0)
        preserve = float(sam.get("bg_l1") or 0) - float(ours.get("bg_l1") or 0)
        scored.append((gap, preserve, task))
    return [task for _, _, task in sorted(scored, reverse=True)[:8]]


def main() -> None:
    make_sheet(SELECTED, "review_selected_methods.jpg")
    make_sheet(hard_keys(), "review_hard_ours_vs_sam.jpg", ["source", "mask", "ours_sd3", "sam_flow_sd3", "flowedit_sd3", "splitflow_sd3", "otrf_enh_sd3"])


if __name__ == "__main__":
    main()
