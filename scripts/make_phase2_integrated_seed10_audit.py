from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REMOTE_PROJECT = Path("/cluster/users/grad/2025/25t8103/project")
PROJECT = REMOTE_PROJECT if REMOTE_PROJECT.exists() else Path(r"I:\Downloads\1\2")
EXP = PROJECT / "experiments" / "support_v3_2026-06-02"
SELECTED = EXP / "phase2_final_selected_runs_2026-06-11.csv"
OUT = PROJECT / "paper" / "assets" / "appendix_grids" / "phase2_integrated_seed10_audit.png"

TASKS = [
    ("T1", "cat_crown", "Cat crown"),
    ("T1", "dog_bow_tie_phase2", "Dog bow tie"),
    ("T1", "dog_front_sunglasses_phase2", "Dog sunglasses"),
    ("T2", "bowl_apple_inside", "Bowl + apple"),
    ("T2", "white_bowl_orange_tabletop_phase2", "White bowl + orange"),
    ("T2", "brown_bowl_lemon_phase2", "Brown bowl + lemon"),
    ("T3", "tshirt_star", "T-shirt star"),
    ("T3", "mug_heart", "Mug heart"),
    ("T3", "tote_leaf", "Tote leaf"),
    ("T4", "red_office_chair_to_blue_office_chair", "Chair blue"),
    ("T4", "green_mug_orange_phase2", "Mug orange"),
    ("T4", "yellow_vase_blue_phase2", "Vase blue"),
    ("T5", "pillow_same_color_cable_knit", "White pillow knit"),
    ("T5", "pillow_same_color_cable_knit_grey", "Grey pillow knit"),
    ("T5", "pillow_same_color_cable_knit_armchair", "Armchair pillow knit"),
]

METHODS = [
    ("source", "Source"),
    ("direct_target", "Direct target"),
    ("adaptive_full_generic_support", "Generic support"),
    ("support_v3_controller_rmsgap", "DeCE-RF"),
    ("sam_flow_sd3", "Sam-Flow SD3"),
    ("sam_flow_flux", "Sam-Flow FLUX"),
]


def read_selected() -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    result_by_key: dict[tuple[str, str], str] = {}
    source_by_task: dict[str, str] = {}
    with SELECTED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("seed") != "10":
                continue
            task = row.get("task", "")
            method = row.get("method", "")
            result = row.get("selected_result_image", "")
            source = row.get("selected_source_image", "")
            if task and source:
                source_by_task.setdefault(task, source)
            if task and method and result:
                result_by_key[(task, method)] = result
    return result_by_key, source_by_task


def resolve(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT / path


def fit_image(path: Path | None, size: int, label: str = "missing") -> Image.Image:
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)
    if not path or not path.exists():
        draw.rectangle((0, 0, size - 1, size - 1), outline=(190, 190, 190))
        draw.text((12, size // 2 - 8), label, fill=(150, 0, 0))
        return canvas
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    draw.rectangle((0, 0, size - 1, size - 1), outline=(220, 220, 220))
    return canvas


def main() -> int:
    results, sources = read_selected()
    thumb = 154
    label_w = 210
    header_h = 52
    pad = 10
    row_gap = 12
    width = label_w + len(METHODS) * (thumb + pad) + pad
    height = header_h + len(TASKS) * (thumb + row_gap) + pad
    out = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    draw.text((pad, 12), "Phase2 seed10 integrated visual audit", fill=(0, 0, 0), font=font)
    for col, (_, label) in enumerate(METHODS):
        x = label_w + col * (thumb + pad)
        draw.text((x, 32), label, fill=(0, 0, 0), font=font)

    y = header_h
    last_family = ""
    for family, task, label in TASKS:
        if family != last_family:
            draw.text((pad, y + 4), family, fill=(80, 80, 80), font=font)
            last_family = family
        draw.text((pad + 34, y + thumb // 2 - 10), label, fill=(0, 0, 0), font=font)
        for col, (method, _) in enumerate(METHODS):
            x = label_w + col * (thumb + pad)
            if method == "source":
                path = resolve(sources.get(task, ""))
            else:
                path = resolve(results.get((task, method), ""))
            out.paste(fit_image(path, thumb), (x, y))
        y += thumb + row_gap

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
