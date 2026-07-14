from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_t4_recolor_19.json"
BASE_METRICS = PROJ / "experiments/flowedit135_fixedmask_metrics_20260623/metrics.csv"
NEW_METRICS = PROJ / "experiments/t4_texture_metrics/metrics.csv"
MASK_DIR = PROJ / "data/flowedit_compatible_135/eval_masks"
OUT_ROWS = PROJ / "experiments/t4_texture_metrics/t4_color_baseline_compare_16_rows.csv"
OUT_SUMMARY = PROJ / "experiments/t4_texture_metrics/t4_color_baseline_compare_16.csv"

EXCLUDE = {
    "fe_017_boat_silhouette_1_sailboat_white_sail_red_hull",
    "fe_029_butterfly_2_red_flower",
    "fe_108_duck_1_colorful_duck",
}
COLOR_RGB = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "red": (200, 30, 30),
    "orange": (230, 130, 20),
    "yellow": (235, 215, 40),
    "gold": (212, 175, 55),
    "golden": (212, 175, 55),
    "green": (40, 160, 60),
    "blue": (40, 90, 200),
    "cyan": (40, 200, 210),
    "teal": (30, 150, 150),
    "purple": (130, 50, 160),
    "violet": (140, 80, 200),
    "pink": (235, 130, 180),
    "brown": (110, 70, 40),
    "beige": (220, 200, 160),
    "silver": (190, 190, 195),
}
COLORS = tuple(COLOR_RGB)
ORDER = [
    "reflex",
    "flowedit_flux",
    "fireflow",
    "ours_flux",
    "rf_solver_edit",
    "flowedit_sd3",
    "splitflow_sd3",
    "ours_sd3",
    "otrf_enh_sd3",
    "drfs_sd3",
    "instruct_pix2pix",
    "ledits_pp",
]
LABEL = {
    "ours_sd3": "Ours-SD3(texture)",
    "ours_flux": "Ours-FLUX(texture)",
    "flowedit_sd3": "FlowEdit-SD3",
    "splitflow_sd3": "SplitFlow-SD3",
    "otrf_enh_sd3": "OT-RF enhanced-SD3",
    "drfs_sd3": "DRFS-SD3",
    "flowedit_flux": "FlowEdit-FLUX",
    "fireflow": "FireFlow",
    "reflex": "ReFLEx",
    "rf_solver_edit": "RF-Solver-Edit",
    "instruct_pix2pix": "InstructPix2Pix",
    "ledits_pp": "LEDITS++",
}


def colors_in(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return [color for color in COLORS if re.search(r"\b" + re.escape(color) + r"\b", text)]


def src_tgt(entry: dict) -> tuple[str | None, str | None]:
    if entry.get("key") == "fe_199_piece_of_cake_2_red_velvet_cake":
        return "brown", "red"
    src = colors_in(entry["source_prompt"])
    tgt = colors_in(entry["target_prompt"])
    source = src[0] if src else None
    target = None
    for color in tgt:
        if color not in set(src):
            target = color
            break
    return source, target or (tgt[-1] if tgt else None)


def lab_of(name: str) -> np.ndarray:
    rgb = np.array(COLOR_RGB[name], dtype=float).reshape(1, 1, 3) / 255.0
    return rgb2lab(rgb)[0, 0]


def read_metrics(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


manifest = [entry for entry in json.loads(MANIFEST.read_text(encoding="utf-8")) if entry["key"] not in EXCLUDE]
tasks = {entry["key"] for entry in manifest}

paths: dict[tuple[str, str], Path] = {}
for row in read_metrics(BASE_METRICS):
    if row.get("task") in tasks and row.get("seed") == "10" and row.get("method") in ORDER:
        paths[(row["task"], row["method"])] = Path(row["result_image"])
for row in read_metrics(NEW_METRICS):
    if row.get("task") in tasks and row.get("seed") == "10" and row.get("method") in {"ours_sd3", "ours_flux"}:
        paths[(row["task"], row["method"])] = Path(row["result_image"])

rows = []
summary: dict[str, dict[str, list[float] | int]] = {
    method: {"S_color": [], "dE_improve": [], "n": 0} for method in ORDER
}
for entry in manifest:
    key = entry["key"]
    source_color, target_color = src_tgt(entry)
    if source_color is None or target_color is None:
        continue
    source = Image.open(entry["image"]).convert("RGB")
    width, height = source.size
    mask_path = MASK_DIR / f"{key}_eval_mask.png"
    if not mask_path.exists():
        continue
    region = np.asarray(Image.open(mask_path).convert("L").resize((width, height))) > 127
    if int(region.sum()) < 50:
        continue
    source_lab = rgb2lab(np.asarray(source, dtype=float) / 255.0)
    source_anchor = lab_of(source_color)
    target_anchor = lab_of(target_color)
    axis = target_anchor - source_anchor
    axis_unit = axis / (np.linalg.norm(axis) + 1e-6)
    de_source = np.linalg.norm(source_lab[region] - target_anchor, axis=1).mean()
    for method in ORDER:
        result_path = paths.get((key, method))
        if result_path is None or not result_path.exists():
            continue
        result = Image.open(result_path).convert("RGB").resize((width, height))
        result_lab = rgb2lab(np.asarray(result, dtype=float) / 255.0)
        delta = result_lab[region] - source_lab[region]
        s_color = float((delta @ axis_unit).mean())
        de_result = np.linalg.norm(result_lab[region] - target_anchor, axis=1).mean()
        de_improve = float(de_source - de_result)
        rows.append(
            {
                "task": key,
                "method": method,
                "display": LABEL.get(method, method),
                "source_color": source_color,
                "target_color": target_color,
                "S_color": s_color,
                "dE_improve": de_improve,
            }
        )
        summary[method]["S_color"].append(s_color)  # type: ignore[index]
        summary[method]["dE_improve"].append(de_improve)  # type: ignore[index]
        summary[method]["n"] = int(summary[method]["n"]) + 1

OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["task", "method", "display", "source_color", "target_color", "S_color", "dE_improve"])
    writer.writeheader()
    writer.writerows(rows)

summary_rows = []
for method in ORDER:
    data = summary[method]
    s_values = data["S_color"]  # type: ignore[assignment]
    de_values = data["dE_improve"]  # type: ignore[assignment]
    if not s_values:
        continue
    summary_rows.append(
        {
            "method": method,
            "display": LABEL.get(method, method),
            "n": int(data["n"]),
            "S_color": float(np.mean(s_values)),
            "dE_improve": float(np.mean(de_values)),
        }
    )

with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["method", "display", "n", "S_color", "dE_improve"])
    writer.writeheader()
    writer.writerows(summary_rows)

print(OUT_SUMMARY)
for row in sorted(summary_rows, key=lambda item: -float(item["S_color"])):
    print(row["display"], row["n"], f"S_color={row['S_color']:.2f}", f"dE={row['dE_improve']:.2f}")
