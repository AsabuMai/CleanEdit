from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"

OUT_AUDIT = EXP / "final_tables_audit_2026-06-11.json"
OUT_MD = EXP / "final_paper_tables_2026-06-11.md"
OUT_TABLE1 = EXP / "table1_core5_final.csv"
OUT_TABLE2A = EXP / "table2a_sd3_common_subset_final.csv"
OUT_TABLE2B = EXP / "table2b_native_context_final.csv"
OUT_TABLE3 = EXP / "table3_phase2_breadth_internal_final.csv"
OUT_TABLE3_FAMILY = EXP / "table3_phase2_breadth_by_family_final.csv"

SEEDS = ["10", "11", "12"]
CORE5_TASKS = [
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_same_color_cable_knit",
]
PHASE2_FAMILIES = {
    "T1_attached_accessory": [
        "cat_crown",
        "dog_bow_tie_phase2",
        "dog_front_sunglasses_phase2",
    ],
    "T2_container_insertion": [
        "bowl_apple_inside",
        "white_bowl_orange_tabletop_phase2",
        "brown_bowl_lemon_phase2",
    ],
    "T3_surface_decal": [
        "tshirt_star",
        "mug_heart",
        "tote_leaf",
    ],
    "T4_local_recolor": [
        "red_office_chair_to_blue_office_chair",
        "green_mug_orange_phase2",
        "yellow_vase_blue_phase2",
    ],
    "T5_same_color_material": [
        "pillow_same_color_cable_knit",
        "pillow_same_color_cable_knit_grey",
        "pillow_same_color_cable_knit_armchair",
    ],
}
PHASE2_TASKS = [task for tasks in PHASE2_FAMILIES.values() for task in tasks]

TABLE1_METHODS = [
    "base_only",
    "direct_target",
    "adaptive_full_generic_support",
    "support_v3_controller_rmsgap",
]
TABLE2A_METHODS = [
    "direct_target",
    "adaptive_full_generic_support",
    "flowedit",
    "flowalign",
    "splitflow",
    "support_v3_fixed",
    "support_v3_controller_rmsgap",
]
TABLE2B_METHODS = [
    "fireflow",
    "rf_solver_edit",
    "reflex",
]
TABLE3_METHODS = TABLE1_METHODS

METHOD_LABELS = {
    "base_only": "RF reconstruction",
    "direct_target": "Direct target",
    "adaptive_full_generic_support": "Generic support",
    "support_v3_fixed": "Fixed CleanEdit-SD3",
    "support_v3_controller_rmsgap": "CleanEdit-SD3",
    "flowedit": "FlowEdit-SD3",
    "flowalign": "FlowAlign-SD3",
    "splitflow": "SplitFlow-SD3",
    "fireflow": "FireFlow-FLUX/context",
    "rf_solver_edit": "RF-Solver-Edit-FLUX/context",
    "reflex": "ReFlex-FLUX/context",
}
METRICS = [
    "outside_mask_l1",
    "inside_mask_l1",
    "source_ssim_luma",
    "dino_source_similarity",
    "edit_score",
    "clip_target_minus_source",
    "runtime_seconds",
    "peak_gpu_memory_gb",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if not any(value == key for key, value in row.items())]


def row_method(row: dict[str, str]) -> str:
    return row.get("method", "") or row.get("baseline", "")


def row_seed(row: dict[str, str]) -> str:
    return row.get("seed", "").removeprefix("seed_")


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", "nan", "None", None}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def avg_values(rows: list[dict[str, str]], key: str) -> float | None:
    values = [value for row in rows if (value := as_float(row, key)) is not None]
    return mean(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def load_sources(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        source_rows = read_csv(path)
        for row in source_rows:
            row["_source_file"] = path.name
        rows.extend(source_rows)
    return rows


def filter_rows(
    rows: list[dict[str, str]],
    tasks: list[str],
    methods: list[str],
    seeds: list[str],
) -> list[dict[str, str]]:
    task_set = set(tasks)
    method_set = set(methods)
    seed_set = set(seeds)
    return [
        row
        for row in rows
        if row.get("task", "") in task_set
        and row_method(row) in method_set
        and row_seed(row) in seed_set
    ]


def audit_collection(
    name: str,
    rows: list[dict[str, str]],
    tasks: list[str],
    methods: list[str],
    seeds: list[str],
    required_metrics: list[str],
) -> dict[str, object]:
    expected = {(task, method, seed) for task in tasks for method in methods for seed in seeds}
    seen: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("task", ""), row_method(row), row_seed(row))
        if key in expected:
            seen[key].append(row)
    missing = sorted(expected - set(seen))
    duplicates = sorted(key for key, values in seen.items() if len(values) > 1)
    missing_metrics = {
        metric: sum(
            1
            for key, values in seen.items()
            for row in values[:1]
            if as_float(row, metric) is None
        )
        for metric in required_metrics
    }
    complete = not missing and not duplicates and not any(missing_metrics.values())
    return {
        "name": name,
        "status": "complete" if complete else "incomplete",
        "expected_rows": len(expected),
        "found_pairs": len(seen),
        "row_count": len(rows),
        "missing_pairs": missing,
        "duplicate_pairs": duplicates,
        "missing_metrics": missing_metrics,
    }


def count_missing_metric_for_methods(
    rows: list[dict[str, str]],
    methods: list[str],
    metric: str,
) -> int:
    method_set = set(methods)
    return sum(1 for row in rows if row_method(row) in method_set and as_float(row, metric) is None)


def task_mean_rows(rows: list[dict[str, str]], methods: list[str], tasks: list[str]) -> dict[str, dict[str, float | None]]:
    by_method_task: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method_task[(row_method(row), row.get("task", ""))].append(row)
    out: dict[str, dict[str, float | None]] = {}
    for method in methods:
        method_task_means: dict[str, list[float]] = defaultdict(list)
        row_count = 0
        success_values = 0
        success_true = 0
        for task in tasks:
            task_rows = by_method_task.get((method, task), [])
            row_count += len(task_rows)
            for metric in METRICS:
                value = avg_values(task_rows, metric)
                if value is not None:
                    method_task_means[metric].append(value)
            for row in task_rows:
                if row.get("task_success") not in {"", None}:
                    success_values += 1
                    success_true += row.get("task_success") == "True"
        summary: dict[str, float | None] = {}
        for metric in METRICS:
            values = method_task_means.get(metric, [])
            summary[metric] = mean(values) if len(values) == len(tasks) else None
        summary["n"] = float(row_count)
        summary["task_success_true"] = float(success_true) if success_values else None
        summary["task_success_denominator"] = float(success_values) if success_values else None
        out[method] = summary
    return out


def summary_rows(rows: list[dict[str, str]], methods: list[str], tasks: list[str]) -> list[dict[str, str]]:
    stats = task_mean_rows(rows, methods, tasks)
    output = []
    for method in methods:
        item = stats[method]
        output.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "n": str(int(item.get("n") or 0)),
                "task_success_true": "" if item.get("task_success_true") is None else str(int(item["task_success_true"] or 0)),
                "task_success_denominator": "" if item.get("task_success_denominator") is None else str(int(item["task_success_denominator"] or 0)),
                "outside_l1": fmt(item.get("outside_mask_l1")),
                "inside_l1": fmt(item.get("inside_mask_l1")),
                "source_ssim_luma": fmt(item.get("source_ssim_luma")),
                "dino_source": fmt(item.get("dino_source_similarity")),
                "edit_score": fmt(item.get("edit_score")),
                "clip_delta": fmt(item.get("clip_target_minus_source")),
                "runtime_s": fmt(item.get("runtime_seconds")),
                "peak_mem_gb": fmt(item.get("peak_gpu_memory_gb")),
            }
        )
    return output


def family_rows(rows: list[dict[str, str]], methods: list[str]) -> list[dict[str, str]]:
    output = []
    for family, tasks in PHASE2_FAMILIES.items():
        for row in summary_rows(rows, methods, tasks):
            row = {"family": family, **row}
            output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines)


def main() -> int:
    table1_rows = filter_rows(
        load_sources([EXP / "e1_core5_strict_metrics.csv"]),
        CORE5_TASKS,
        TABLE1_METHODS,
        SEEDS,
    )
    table2a_rows = filter_rows(
        load_sources(
            [
                EXP / "e1_t1_t4_directgeneric_metrics.csv",
                EXP / "e2_t1_t4_baseline_fixed_mask_metrics.csv",
                EXP / "table2a_e4_common_subset_clipdino_metrics.csv",
                EXP / "table2_t5_internal_metrics.csv",
                EXP / "table2_t5_baseline_metrics.csv",
            ]
        ),
        PHASE2_TASKS,
        TABLE2A_METHODS,
        SEEDS,
    )
    table2b_rows = filter_rows(
        load_sources(
            [
                EXP / "e2_t1_t4_baseline_fixed_mask_metrics.csv",
                EXP / "table2_t5_baseline_metrics.csv",
            ]
        ),
        PHASE2_TASKS,
        TABLE2B_METHODS,
        SEEDS,
    )
    table3_rows = filter_rows(
        load_sources(
            [
                EXP / "e4_t1_t4_reconstruction_floor_metrics.csv",
                EXP / "e1_t1_t4_directgeneric_metrics.csv",
                EXP / "table2a_e4_common_subset_clipdino_metrics.csv",
                EXP / "table2_t5_internal_metrics.csv",
            ]
        ),
        PHASE2_TASKS,
        TABLE3_METHODS,
        SEEDS,
    )

    audits = [
        audit_collection("table1_core5", table1_rows, CORE5_TASKS, TABLE1_METHODS, SEEDS, ["outside_mask_l1", "source_ssim_luma"]),
        audit_collection("table2a_sd3_common_subset", table2a_rows, PHASE2_TASKS, TABLE2A_METHODS, SEEDS, ["outside_mask_l1", "source_ssim_luma", "edit_score"]),
        audit_collection("table2b_native_context", table2b_rows, PHASE2_TASKS, TABLE2B_METHODS, SEEDS, ["outside_mask_l1", "source_ssim_luma", "edit_score"]),
        audit_collection("table3_phase2_internal", table3_rows, PHASE2_TASKS, TABLE3_METHODS, SEEDS, ["outside_mask_l1", "source_ssim_luma"]),
    ]
    table3_edit_missing = count_missing_metric_for_methods(
        table3_rows,
        ["direct_target", "adaptive_full_generic_support", "support_v3_controller_rmsgap"],
        "edit_score",
    )
    audits[-1]["non_floor_edit_score_missing"] = table3_edit_missing
    if table3_edit_missing:
        audits[-1]["status"] = "incomplete"
    audit_doc = {
        "status": "complete" if all(item["status"] == "complete" for item in audits) else "incomplete",
        "scope_note": "Final table aggregates use the corrected Phase2 T5 full-pillow task set; E5 removal is excluded from Table 1 and Table 3.",
        "tables": audits,
    }

    table1 = summary_rows(table1_rows, TABLE1_METHODS, CORE5_TASKS)
    table2a = summary_rows(table2a_rows, TABLE2A_METHODS, PHASE2_TASKS)
    table2b = summary_rows(table2b_rows, TABLE2B_METHODS, PHASE2_TASKS)
    table3 = summary_rows(table3_rows, TABLE3_METHODS, PHASE2_TASKS)
    table3_family = family_rows(table3_rows, TABLE3_METHODS)

    write_csv(OUT_TABLE1, table1)
    write_csv(OUT_TABLE2A, table2a)
    write_csv(OUT_TABLE2B, table2b)
    write_csv(OUT_TABLE3, table3)
    write_csv(OUT_TABLE3_FAMILY, table3_family)
    OUT_AUDIT.write_text(json.dumps(audit_doc, indent=2) + "\n", encoding="utf-8")

    fields = ["label", "n", "outside_l1", "inside_l1", "source_ssim_luma", "dino_source", "edit_score", "clip_delta"]
    lines = [
        "# Final Paper Tables (2026-06-11)",
        "",
        "These tables are built from frozen CSV artifacts and the corrected T5 full-pillow task set.",
        "",
        f"Audit status: `{audit_doc['status']}` (`{OUT_AUDIT.relative_to(ROOT)}`).",
        "",
        "## Table 1: Strict Core-5 Main Effect",
        "",
        md_table(table1, ["label", "n", "task_success_true", "task_success_denominator", *fields[2:]]),
        "",
        "Success columns apply to relation-style T1/T2/T3 checks only; T4/T5 use operation-specific metrics and visual/material gates.",
        "",
        "## Table 2a: Same-Backbone SD3 Common Subset",
        "",
        md_table(table2a, fields),
        "",
        "Table 2a is a same-protocol SD3/common-subset comparison. Fixed CleanEdit-SD3 is a component-control row, not an external baseline.",
        "",
        "## Table 2b: Native Preservation-Aware RF / FLUX Context",
        "",
        md_table(table2b, fields),
        "",
        "Table 2b is contextual and cross-backbone; do not use it for algorithm-level superiority claims.",
        "",
        "## Table 3: Phase2 15-Task Breadth Validation",
        "",
        md_table(table3, fields),
        "",
        "Phase2 is breadth/robustness validation over 15 controlled tasks, not a large-scale benchmark. E5 removal probes are excluded. Edit-score completeness is required for editing rows; the RF reconstruction row is a reconstruction floor.",
        "",
        "## Table 3 Family Breakdown",
        "",
        md_table(table3_family, ["family", *fields]),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"status={audit_doc['status']}")
    for item in audits:
        print(
            f"{item['name']}: status={item['status']} "
            f"expected={item['expected_rows']} found_pairs={item['found_pairs']} rows={item['row_count']}"
        )
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0 if audit_doc["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
