from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
OLD = PROJ / "experiments/norestore_metrics/metrics.csv"
NEW_T4 = PROJ / "experiments/t4_texture_metrics/metrics.csv"
OUT_DIR = PROJ / "experiments/t4_texture_metrics"
MERGED = OUT_DIR / "metrics_merged135.csv"
SUMMARY = OUT_DIR / "summary_merged135.csv"

METHODS_TO_REPLACE = {"ours_sd3", "ours_flux"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


old_rows = read_rows(OLD)
new_rows = read_rows(NEW_T4)
new_map = {
    (row["task"], row["method"], row["seed"]): row
    for row in new_rows
    if row.get("method") in METHODS_TO_REPLACE
}

merged = []
replaced = 0
for row in old_rows:
    key = (row.get("task", ""), row.get("method", ""), row.get("seed", ""))
    if key in new_map:
        merged.append(new_map[key])
        replaced += 1
    else:
        merged.append(row)

fieldnames = list(old_rows[0].keys())
for row in merged:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

with MERGED.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged)

metric_cols = [
    "clip_t",
    "clip_target_minus_source",
    "clip_direction_similarity",
    "local_clip_t",
    "local_clip_t_full_prompt",
    "local_clip_target_minus_source",
    "clip_image_source_similarity",
    "dino_source_similarity",
    "bg_dino_source",
    "lpips_full",
    "lpips_outside",
    "bg_lpips",
    "source_l1",
    "inside_l1",
    "outside_l1",
    "bg_l1",
    "source_ssim_luma",
    "bg_ssim_luma",
    "eval_mask_area",
    "dilated_edit_mask_area",
    "bg_mask_area",
]


def vals(items: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for item in items:
        raw = item.get(key, "")
        if raw == "":
            continue
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


groups: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in merged:
    groups[row["method"]].append(row)

summary_rows = []
for method, items in sorted(groups.items()):
    rec = {
        "method": method,
        "method_display": items[0].get("method_display", method),
        "n": len(items),
        "complete_n": sum(str(row.get("complete")) == "True" for row in items),
        "fixed_mask_n": sum(row.get("mask_source") == "fixed_eval_mask" for row in items),
        "local_clip_t_n": sum(1 for row in items if row.get("local_clip_t")),
    }
    for key in metric_cols:
        xs = vals(items, key)
        rec[key] = "" if not xs else sum(xs) / len(xs)
    summary_rows.append(rec)

with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["method", "method_display", "n", "complete_n", "fixed_mask_n", "local_clip_t_n"] + metric_cols,
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("replaced", replaced)
print("merged", MERGED)
print("summary", SUMMARY)
