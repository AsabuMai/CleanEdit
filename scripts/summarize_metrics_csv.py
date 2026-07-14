from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRIC_COLS = [
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


def vals(rows: list[dict[str, str]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def mean(xs: list[float]) -> str | float:
    return "" if not xs else sum(xs) / len(xs)


def std(xs: list[float]) -> str | float:
    if len(xs) <= 1:
        return ""
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(items: list[dict[str, str]], method: str) -> dict[str, object]:
    rec: dict[str, object] = {
        "method": method,
        "method_display": items[0].get("method_display", method),
        "n": len(items),
        "complete_n": sum(str(row.get("complete")) == "True" for row in items),
        "fixed_mask_n": sum(row.get("mask_source") == "fixed_eval_mask" for row in items),
        "local_clip_t_n": sum(1 for row in items if row.get("local_clip_t")),
    }
    for key in METRIC_COLS:
        rec[key] = mean(vals(items, key))
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-per-method", type=int, default=0)
    parser.add_argument("--mean-std", action="store_true")
    args = parser.parse_args()

    metrics = Path(args.metrics)
    out_dir = Path(args.out_dir)
    rows = list(csv.DictReader(metrics.open(newline="", encoding="utf-8")))

    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_family: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_method_seed: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "")
        by_method[method].append(row)
        by_family[(row.get("family_label", ""), method)].append(row)
        by_method_seed[(method, row.get("seed", ""))].append(row)

    base_fields = [
        "method",
        "method_display",
        "n",
        "complete_n",
        "fixed_mask_n",
        "local_clip_t_n",
    ]
    summary = [summarize_group(items, method) for method, items in sorted(by_method.items())]
    write_csv(out_dir / "summary_by_method.csv", base_fields + METRIC_COLS, summary)

    family_fields = ["family_label"] + base_fields + METRIC_COLS
    family_rows: list[dict[str, object]] = []
    for (family, method), items in sorted(by_family.items()):
        rec = summarize_group(items, method)
        rec["family_label"] = family
        family_rows.append(rec)
    write_csv(out_dir / "summary_by_family_method.csv", family_fields, family_rows)

    seed_fields = ["method", "method_display", "seed", "n", "complete_n", "fixed_mask_n", "local_clip_t_n"] + METRIC_COLS
    seed_rows: list[dict[str, object]] = []
    for (method, seed), items in sorted(by_method_seed.items()):
        rec = summarize_group(items, method)
        rec["seed"] = seed
        seed_rows.append(rec)
    write_csv(out_dir / "summary_by_method_seed.csv", seed_fields, seed_rows)

    if args.mean_std:
        mean_std_fields = base_fields[:]
        for key in METRIC_COLS:
            mean_std_fields += [f"{key}_mean", f"{key}_std"]
        mean_std_rows: list[dict[str, object]] = []
        for method, items in sorted(by_method.items()):
            rec: dict[str, object] = {
                "method": method,
                "method_display": items[0].get("method_display", method),
                "n": len(items),
                "complete_n": sum(str(row.get("complete")) == "True" for row in items),
                "fixed_mask_n": sum(row.get("mask_source") == "fixed_eval_mask" for row in items),
                "local_clip_t_n": sum(1 for row in items if row.get("local_clip_t")),
            }
            for key in METRIC_COLS:
                xs = vals(items, key)
                rec[f"{key}_mean"] = mean(xs)
                rec[f"{key}_std"] = std(xs)
            mean_std_rows.append(rec)
        write_csv(out_dir / "summary_by_method_mean_std.csv", mean_std_fields, mean_std_rows)

    bad = []
    if args.expected_per_method:
        for rec in summary:
            if rec["n"] != args.expected_per_method or rec["complete_n"] != args.expected_per_method:
                bad.append(rec)
            if rec["fixed_mask_n"] != args.expected_per_method:
                bad.append(rec)
            if rec["local_clip_t_n"] != args.expected_per_method:
                bad.append(rec)

    audit = {
        "rows": len(rows),
        "methods": sorted(by_method),
        "method_counts": {method: len(items) for method, items in sorted(by_method.items())},
        "bad": bad,
    }
    (out_dir / "metric_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 3 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
