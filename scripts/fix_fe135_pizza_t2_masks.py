from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MASK_DIR = PROJ / "data/flowedit_compatible_135/sam_support_masks_17new"
MANIFEST17 = PROJ / "data/flowedit_compatible_135/manifest_sam_17new.json"
CONTACT = MASK_DIR / "_contact_sheet.jpg"

KEYS = [
    "fe_207_pizza_tomato_olive_1_pepperoni",
    "fe_208_pizza_tomato_olive_2_mushrooms",
]


def fix_one(key: str) -> None:
    meta_path = MASK_DIR / f"{key}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    anchor_box = meta["anchor_box_xyxy"]
    x0, y0, x1, y1 = [float(v) for v in anchor_box]
    width = height = 512
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    scale = 0.84
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale
    ellipse = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).ellipse(ellipse, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    area = float((arr > 0.5).mean())
    ys, xs = np.where(arr > 0.5)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if len(xs) else [0, 0, 0, 0]

    support = MASK_DIR / f"{key}_support.png"
    backup = MASK_DIR / f"{key}_support_bbox_backup.png"
    if support.exists() and not backup.exists():
        support.rename(backup)
    mask.save(support)

    meta["support_area_before_manual_fix"] = meta.get("support_area")
    meta["support_box_xyxy_before_manual_fix"] = meta.get("support_box_xyxy")
    meta["support_mask_before_manual_fix"] = str(backup)
    meta["support_area"] = area
    meta["support_box_xyxy"] = bbox
    meta["support_mask"] = str(support)
    meta["manual_fix"] = {
        "reason": "T2 pizza topping mask used a square support bbox that included table/background.",
        "method": "inner_ellipse_from_anchor_box",
        "scale": scale,
        "blur_radius": 2.0,
        "updated_at_jst": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(key, f"area={area:.3f}", f"bbox={bbox}", flush=True)


def update_audit() -> None:
    audit = MASK_DIR / "_audit.csv"
    rows = list(csv.DictReader(audit.open(encoding="utf-8")))
    fields = list(rows[0].keys())
    for row in rows:
        if row["key"] in KEYS:
            meta = json.loads((MASK_DIR / f"{row['key']}.json").read_text(encoding="utf-8"))
            row["support_area"] = f"{meta['support_area']:.6f}"
            row["support_mask"] = str(MASK_DIR / f"{row['key']}_support.png")
            row["error"] = "manual_inner_ellipse_fix"
    with audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (255, 60, 20, 0))
    red.putalpha(mask.point(lambda v: int(v * 0.62)))
    return Image.alpha_composite(base, red).convert("RGB")


def rebuild_contact() -> None:
    items = json.loads(MANIFEST17.read_text(encoding="utf-8"))
    cols = 8
    thumb = 160
    label_h = 24
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, item in enumerate(items):
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + label_h)
        img = Image.open(item["image"]).convert("RGB").resize((thumb, thumb))
        mask = Image.open(PROJ / item["pp_local_mask"]).convert("L").resize((thumb, thumb))
        sheet.paste(overlay(img, mask), (x, y + label_h))
        draw.text((x + 3, y + 5), item["key"][:23], fill=(0, 0, 0))
    sheet.save(CONTACT, quality=92)
    print("contact", CONTACT, flush=True)


def main() -> None:
    for key in KEYS:
        fix_one(key)
    update_audit()
    rebuild_contact()


if __name__ == "__main__":
    main()
