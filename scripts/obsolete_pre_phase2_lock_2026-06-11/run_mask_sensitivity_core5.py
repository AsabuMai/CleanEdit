from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
MASK_SRC = EXP / "eval_masks"
MASK_ROOT = EXP / "mask_sensitivity_core5_masks"
OUT_AUDIT = EXP / "mask_sensitivity_core5_audit.json"
OUT_SUMMARY = EXP / "mask_sensitivity_core5_summary.csv"
OUT_MD = EXP / "mask_sensitivity_core5_summary.md"

TASKS = [
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_same_color_cable_knit",
]
METHODS = [
    "base_only",
    "direct_target",
    "adaptive_full_generic_support",
    "support_v3_controller_rmsgap",
]
SEEDS = ["10", "11", "12"]
VARIANTS = {
    "eroded": ("min", 15),
    "base": ("copy", 1),
    "dilated": ("max", 15),
}
LABELS = {
    "base_only": "RF reconstruction",
    "direct_target": "Direct target",
    "adaptive_full_generic_support": "Generic support",
    "support_v3_controller_rmsgap": "DeCE-RF-SD3",
}


def mask_area(path: Path) -> float:
    image = Image.open(path).convert("L")
    pixels = list(image.getdata())
    return sum(value > 51 for value in pixels) / max(1, len(pixels))


def build_masks() -> dict[str, dict[str, float]]:
    areas: dict[str, dict[str, float]] = {variant: {} for variant in VARIANTS}
    for variant, (op, kernel) in VARIANTS.items():
        out_dir = MASK_ROOT / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            src = MASK_SRC / f"{task}_eval_mask.png"
            dst = out_dir / f"{task}_eval_mask.png"
            if op == "copy":
                shutil.copyfile(src, dst)
            else:
                image = Image.open(src).convert("L")
                if op == "min":
                    image = image.filter(ImageFilter.MinFilter(kernel))
                elif op == "max":
                    image = image.filter(ImageFilter.MaxFilter(kernel))
                image.save(dst)
            areas[variant][task] = mask_area(dst)
    return areas


def run_metrics(variant: str) -> Path:
    csv_out = EXP / f"mask_sensitivity_core5_{variant}_metrics.csv"
    json_out = EXP / f"mask_sensitivity_core5_{variant}_metrics.json"
    cmd = [
        sys.executable,
        "scripts/evaluate_paper_metrics.py",
        "--outputs-dir",
        str(ROOT / "outputs" / "pretty_matrix"),
        "--csv-output",
        str(csv_out),
        "--json-output",
        str(json_out),
        "--task-names",
        " ".join(TASKS),
        "--method-names",
        " ".join(METHODS),
        "--seeds",
        " ".join(SEEDS),
        "--eval-mask-dir",
        str(MASK_ROOT / variant),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return csv_out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def avg(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return mean(values) if values else float("nan")


def summarize(metrics_paths: dict[str, Path]) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    rows_out: list[dict[str, str]] = []
    rankings: dict[str, list[str]] = {}
    for variant, path in metrics_paths.items():
        rows = read_csv(path)
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["method"]].append(row)
        variant_summary = []
        for method in METHODS:
            method_rows = grouped[method]
            variant_summary.append(
                {
                    "variant": variant,
                    "method": method,
                    "label": LABELS[method],
                    "n": str(len(method_rows)),
                    "outside_l1": f"{avg(method_rows, 'outside_mask_l1'):.4f}",
                    "inside_l1": f"{avg(method_rows, 'inside_mask_l1'):.4f}",
                    "source_ssim_luma": f"{avg(method_rows, 'source_ssim_luma'):.4f}",
                }
            )
        variant_summary.sort(key=lambda row: float(row["outside_l1"]))
        rankings[variant] = [row["method"] for row in variant_summary]
        rows_out.extend(variant_summary)
    return rows_out, rankings


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, str]]) -> str:
    fields = ["variant", "label", "n", "outside_l1", "inside_l1", "source_ssim_luma"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines)


def main() -> int:
    areas = build_masks()
    metrics_paths = {variant: run_metrics(variant) for variant in VARIANTS}
    summary, rankings = summarize(metrics_paths)
    write_csv(OUT_SUMMARY, summary)

    expected_rows = len(TASKS) * len(METHODS) * len(SEEDS)
    metric_row_counts = {
        variant: len(read_csv(path))
        for variant, path in metrics_paths.items()
    }
    mask_area_ok = all(area > 0.0001 for variant in areas.values() for area in variant.values())
    row_counts_ok = all(count == expected_rows for count in metric_row_counts.values())
    base_order = rankings["base"]
    full_order_stable = all(order == base_order for order in rankings.values())
    top_method_stable = all(order and order[0] == base_order[0] for order in rankings.values())
    audit = {
        "status": "complete" if (mask_area_ok and row_counts_ok and full_order_stable) else "incomplete",
        "expected_rows_per_variant": expected_rows,
        "metric_row_counts": metric_row_counts,
        "mask_areas": areas,
        "rankings_by_outside_l1": rankings,
        "full_order_stable": full_order_stable,
        "top_method_stable": top_method_stable,
        "ranking_reference": "outside_mask_l1 ascending",
        "mask_kernel_size": 15,
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Core-5 Mask Sensitivity Sweep",
        "",
        f"Audit status: `{audit['status']}` (`{OUT_AUDIT.relative_to(ROOT)}`).",
        "",
        "Evaluation masks are eroded/base/dilated with a 15x15 min/max filter. Ranking uses mean outside-mask L1, ascending.",
        "",
        f"Full order stable: `{full_order_stable}`. Top method stable: `{top_method_stable}`.",
        "",
        md_table(summary),
        "",
        "Rankings:",
    ]
    for variant, order in rankings.items():
        lines.append(f"- `{variant}`: " + " > ".join(LABELS.get(method, method) for method in order))
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"status={audit['status']}")
    print(f"row_counts={metric_row_counts}")
    print(f"rankings={rankings}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0 if audit["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
