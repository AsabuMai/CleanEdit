from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


PROJ = Path(__file__).resolve().parents[1]
T4_MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_t4_recolor_19.json"
BASE_METRICS = PROJ / "experiments/flowedit135_fixedmask_metrics_20260623/metrics.csv"
NEW_METRICS = PROJ / "experiments/t4_texture_metrics/metrics.csv"
OUT = PROJ / "experiments/t4_texture_metrics/t4_baseline_compare_19.csv"

T4 = {entry["key"] for entry in json.loads(T4_MANIFEST.read_text(encoding="utf-8"))}
BASE_ROWS = list(csv.DictReader(BASE_METRICS.open(newline="", encoding="utf-8")))
NEW_ROWS = list(csv.DictReader(NEW_METRICS.open(newline="", encoding="utf-8")))
REPLACE = {
    (row["task"], row["method"], row["seed"]): row
    for row in NEW_ROWS
    if row["method"] in {"ours_sd3", "ours_flux"}
}

rows: list[dict[str, str]] = []
for row in BASE_ROWS:
    if row.get("task") not in T4 or row.get("seed") != "10":
        continue
    key = (row["task"], row["method"], row["seed"])
    rows.append(REPLACE.get(key, row))

seen = {(row["task"], row["method"], row["seed"]) for row in rows}
for key, row in REPLACE.items():
    if key not in seen:
        rows.append(row)

cols = [
    "local_clip_t",
    "clip_direction_similarity",
    "clip_target_minus_source",
    "local_clip_target_minus_source",
    "bg_lpips",
    "bg_l1",
    "bg_dino_source",
    "bg_ssim_luma",
]
labels = {
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
order = [
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

by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in rows:
    by_method[row["method"]].append(row)


def mean(items: list[dict[str, str]], key: str) -> float | str:
    values = []
    for row in items:
        raw = row.get(key, "")
        if raw == "":
            continue
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return "" if not values else sum(values) / len(values)


summary = []
for method in order:
    if method not in by_method:
        continue
    rec: dict[str, str | float | int] = {
        "method": method,
        "display": labels.get(method, method),
        "n": len(by_method[method]),
    }
    for col in cols:
        rec[col] = mean(by_method[method], col)
    summary.append(rec)

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["method", "display", "n"] + cols)
    writer.writeheader()
    writer.writerows(summary)

print(OUT)
for row in summary:
    print(
        row["display"],
        row["n"],
        "CLIP-T",
        f"{row['local_clip_t']:.4f}",
        "Dir",
        f"{row['clip_direction_similarity']:.4f}",
        "Margin",
        f"{row['clip_target_minus_source']:.4f}",
        "BG-LPIPS",
        f"{row['bg_lpips']:.4f}",
        "BG-L1",
        f"{row['bg_l1']:.4f}",
    )
