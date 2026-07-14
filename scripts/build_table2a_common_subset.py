from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"

OUT_CSV = EXP / "table2a_sd3_t1_t4_common_subset_prelim.csv"
OUT_AUDIT = EXP / "table2a_sd3_t1_t4_common_subset_prelim_audit.json"
OUT_MD = EXP / "table2a_sd3_t1_t4_common_subset_prelim_2026-06-11.md"

TASKS = [
    "cat_crown",
    "dog_bow_tie_phase2",
    "dog_front_sunglasses_phase2",
    "bowl_apple_inside",
    "white_bowl_orange_tabletop_phase2",
    "brown_bowl_lemon_phase2",
    "tshirt_star",
    "mug_heart",
    "tote_leaf",
    "red_office_chair_to_blue_office_chair",
    "green_mug_orange_phase2",
    "yellow_vase_blue_phase2",
]
SEEDS = ["10", "11", "12"]

METHOD_SOURCES = {
    "direct_target": "e1_t1_t4_directgeneric_metrics.csv",
    "adaptive_full_generic_support": "e1_t1_t4_directgeneric_metrics.csv",
    "flowedit": "e2_t1_t4_baseline_fixed_mask_metrics.csv",
    "flowalign": "e2_t1_t4_baseline_fixed_mask_metrics.csv",
    "splitflow": "e2_t1_t4_baseline_fixed_mask_metrics.csv",
    "sam_flow_sd3": "e2_t1_t4_baseline_fixed_mask_metrics.csv",
    "support_v3_fixed": "table2a_e4_common_subset_clipdino_metrics.csv",
    "support_v3_controller_rmsgap": "table2a_e4_common_subset_clipdino_metrics.csv",
}

METHOD_LABELS = {
    "direct_target": "Direct target",
    "adaptive_full_generic_support": "Generic support",
    "flowedit": "FlowEdit-SD3",
    "flowalign": "FlowAlign-SD3",
    "splitflow": "SplitFlow-SD3",
    "sam_flow_sd3": "Sam-Flow-SD3",
    "support_v3_fixed": "Fixed CleanEdit-SD3",
    "support_v3_controller_rmsgap": "CleanEdit-SD3",
}

ORDER = [
    "direct_target",
    "adaptive_full_generic_support",
    "flowedit",
    "flowalign",
    "splitflow",
    "sam_flow_sd3",
    "support_v3_fixed",
    "support_v3_controller_rmsgap",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if not any(v == k for k, v in r.items())]


def to_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", "nan", "None", None}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def avg(rows: list[dict[str, str]], key: str) -> str:
    vals = [v for row in rows if (v := to_float(row, key)) is not None]
    return f"{mean(vals):.4f}" if vals else ""


def load_method_rows() -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, object]]]:
    by_method: dict[str, list[dict[str, str]]] = {}
    audit: dict[str, dict[str, object]] = {}
    cache: dict[str, list[dict[str, str]]] = {}
    expected_pairs = {(task, seed) for task in TASKS for seed in SEEDS}

    for method, filename in METHOD_SOURCES.items():
        path = EXP / filename
        if not path.exists():
            by_method[method] = []
            audit[method] = {
                "source": filename,
                "status": "missing_source",
                "missing_pairs": sorted(expected_pairs),
                "row_count": 0,
            }
            continue
        if filename not in cache:
            cache[filename] = read_rows(path)
        rows = [
            row
            for row in cache[filename]
            if row.get("method") == method
            and row.get("task") in TASKS
            and row.get("seed", "").removeprefix("seed_") in SEEDS
        ]
        pairs = {(row.get("task", ""), row.get("seed", "").removeprefix("seed_")) for row in rows}
        missing_pairs = sorted(expected_pairs - pairs)
        duplicate_count = len(rows) - len(pairs)
        missing_metrics = {
            key: sum(1 for row in rows if to_float(row, key) is None)
            for key in ["dino_source_similarity", "edit_score", "clip_target_minus_source"]
        }
        status = "complete"
        if missing_pairs:
            status = "missing_pairs"
        elif duplicate_count:
            status = "duplicate_pairs"
        elif any(missing_metrics.values()):
            status = "missing_metrics"
        by_method[method] = rows
        audit[method] = {
            "source": filename,
            "status": status,
            "row_count": len(rows),
            "pair_count": len(pairs),
            "missing_pairs": missing_pairs,
            "duplicate_count": duplicate_count,
            "missing_metrics": missing_metrics,
        }
    return by_method, audit


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    fields = [
        "label",
        "n",
        "outside_l1",
        "inside_l1",
        "source_ssim_luma",
        "dino_source",
        "edit_score",
        "clip_delta",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines)


def main() -> None:
    by_method, audit = load_method_rows()
    expected_n = len(TASKS) * len(SEEDS)
    complete = all(audit[method]["status"] == "complete" for method in ORDER)

    summary = []
    for method in ORDER:
        rows = by_method[method]
        summary.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "n": str(len(rows)),
                "expected_n": str(expected_n),
                "source": METHOD_SOURCES[method],
                "outside_l1": avg(rows, "outside_mask_l1"),
                "inside_l1": avg(rows, "inside_mask_l1"),
                "source_ssim_luma": avg(rows, "source_ssim_luma"),
                "dino_source": avg(rows, "dino_source_similarity"),
                "edit_score": avg(rows, "edit_score"),
                "clip_delta": avg(rows, "clip_target_minus_source"),
                "runtime_s": avg(rows, "runtime_seconds"),
                "peak_mem_gb": avg(rows, "peak_gpu_memory_gb"),
            }
        )

    audit_doc = {
        "status": "complete_t1_t4_preliminary" if complete else "incomplete",
        "scope": "Phase2 T1-T4 only; not final Table 2 until Phase2 T5 is complete.",
        "task_count": len(TASKS),
        "seed_count": len(SEEDS),
        "expected_rows_per_method": expected_n,
        "tasks": TASKS,
        "seeds": SEEDS,
        "methods": audit,
    }

    write_csv(OUT_CSV, summary)
    OUT_AUDIT.write_text(json.dumps(audit_doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Table 2a SD3 T1-T4 Common-Subset Preliminary",
        "",
        "Common subset: Phase2 T1-T4 task families, 12 tasks x seeds 10/11/12 = 36 rows per method.",
        "",
        "Not final Table 2: Phase2 T5 material-replacement rows are not complete for baseline methods yet.",
        "",
        f"Audit status: `{audit_doc['status']}`.",
        "",
        markdown_table(summary),
        "",
        "Sources:",
        "",
        "- Direct target / Generic support: `e1_t1_t4_directgeneric_metrics.csv`.",
        "- FlowEdit / FlowAlign / SplitFlow: `e2_t1_t4_baseline_fixed_mask_metrics.csv`.",
        "- Fixed CleanEdit / CleanEdit: `table2a_e4_common_subset_clipdino_metrics.csv`.",
        "",
        "Use this only as a preliminary T1-T4 diagnostic. Final Table 2 must be rebuilt after T5 is complete.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"status={audit_doc['status']}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_AUDIT}")
    print(f"wrote {OUT_MD}")
    if not complete:
        for method in ORDER:
            item = audit[method]
            if item["status"] != "complete":
                print(f"{method}: {item['status']} source={item['source']}")


if __name__ == "__main__":
    main()
