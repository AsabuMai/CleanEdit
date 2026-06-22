from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
EXP = PROJ / "experiments/flowedit135_fixedmask_metrics_20260621"
RUNS = PROJ / "outputs/flowedit135_metric_runs"
MASK_DIR = PROJ / "data/flowedit_compatible_135/eval_masks"
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest.json"
METRICS = EXP / "metrics.csv"

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
]


def font(size: int):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_SMALL = font(11)
F_LABEL = font(12)


def load_manifest() -> list[dict]:
    return json.load(MANIFEST.open(encoding="utf-8"))


def source_image(item: dict) -> Image.Image:
    return Image.open(item["image"]).convert("RGB")


def mask_overlay(item: dict) -> Image.Image:
    base = source_image(item).convert("RGBA")
    mask = Image.open(MASK_DIR / f"{item['key']}_eval_mask.png").convert("L").resize(base.size)
    red = Image.new("RGBA", base.size, (255, 85, 35, 0))
    red.putalpha(mask.point(lambda v: int(v * 0.42)))
    return Image.alpha_composite(base, red).convert("RGB")


def method_image(item: dict, method: str) -> Image.Image:
    key = item["key"]
    if method == "source":
        return source_image(item)
    if method == "mask":
        return mask_overlay(item)
    return Image.open(RUNS / key / method / "seed_10/result.png").convert("RGB")


def thumb(img: Image.Image, w: int, h: int) -> Image.Image:
    canvas = Image.new("RGB", (w, h), "white")
    img = img.convert("RGB")
    img.thumbnail((w, h - 18))
    canvas.paste(img, ((w - img.width) // 2, 0))
    return canvas


def metric_index() -> dict[tuple[str, str], dict]:
    rows = csv.DictReader(METRICS.open(newline="", encoding="utf-8"))
    return {(r["task"], r["method"]): r for r in rows}


def draw_sheet(family: str, items: list[dict], metrics: dict[tuple[str, str], dict]) -> Path:
    cell_w, cell_h = 128, 124
    label_w = 260
    header_h = 34
    w = label_w + cell_w * len(METHODS)
    h = header_h + cell_h * len(items)
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), f"{family} ({len(items)} tasks)", fill=(0, 0, 0), font=F_LABEL)
    for c, method in enumerate(METHODS):
        draw.text((label_w + c * cell_w + 4, 8), method, fill=(0, 0, 0), font=F_SMALL)
    for r, item in enumerate(items):
        y = header_h + r * cell_h
        key = item["key"]
        ours = metrics.get((key, "ours_sd3"), {})
        sam = metrics.get((key, "sam_flow_sd3"), {})
        draw.text((6, y + 4), key[:38], fill=(0, 0, 0), font=F_SMALL)
        draw.text((6, y + 20), item["target_prompt"][:44], fill=(45, 45, 45), font=F_SMALL)
        if ours:
            txt = f"ours local {float(ours['local_clip_t']):.3f} bgL1 {float(ours['bg_l1']):.4f}"
            draw.text((6, y + 38), txt, fill=(180, 80, 20), font=F_SMALL)
        if sam:
            txt = f"sam local {float(sam['local_clip_t']):.3f} bgL1 {float(sam['bg_l1']):.4f}"
            draw.text((6, y + 54), txt, fill=(50, 90, 150), font=F_SMALL)
        for c, method in enumerate(METHODS):
            x = label_w + c * cell_w
            try:
                img = method_image(item, method)
                panel = thumb(img, cell_w, cell_h)
            except Exception:
                panel = Image.new("RGB", (cell_w, cell_h), (238, 238, 238))
            sheet.paste(panel, (x, y))
        draw.line([(0, y), (w, y)], fill=(235, 235, 235))
    out = EXP / f"family_review_{family}.jpg"
    sheet.save(out, quality=92)
    return out


def main() -> None:
    manifest = load_manifest()
    metrics = metric_index()
    families = {}
    for item in manifest:
        families.setdefault(item.get("family_label", "unknown"), []).append(item)
    for family, items in sorted(families.items()):
        out = draw_sheet(family, items, metrics)
        print(out)


if __name__ == "__main__":
    main()
