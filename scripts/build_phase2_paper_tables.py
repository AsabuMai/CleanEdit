from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"

OUT_AUDIT = EXP / "phase2_tables_audit_2026-06-11.json"
OUT_MD = EXP / "phase2_paper_tables_2026-06-11.md"
OUT_TASK_MAP = EXP / "phase2_t1_t5_task_map_2026-06-11.csv"
OUT_TABLE1 = EXP / "table1_phase2_t1_t5_main_final.csv"
OUT_TABLE2A = EXP / "table2a_phase2_sd3_common_subset_final.csv"
OUT_TABLE2B = EXP / "table2b_phase2_native_context_final.csv"
OUT_FAMILY = EXP / "phase2_t1_t5_family_breakdown_final.csv"
INTERNAL_BG_METRICS = EXP / "phase2_internal_bg_metrics.csv"

SEEDS = ["10", "11", "12"]
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
TABLE2B_EXCLUDE = {"white_bowl_orange_tabletop_phase2"}  # FLUX-intrinsic failure; dropped from native FLUX comparison
TABLE2B_TASKS = [t for t in PHASE2_TASKS if t not in TABLE2B_EXCLUDE]

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
    "splitflow",
    "sam_flow_sd3",
    "support_v3_fixed",
    "support_v3_controller_rmsgap",
]
TABLE2B_METHODS = ["fireflow", "rf_solver_edit", "reflex", "sam_flow_flux", "dece_rf_flux"]

METHOD_LABELS = {
    "base_only": "RF reconstruction",
    "direct_target": "Direct target",
    "adaptive_full_generic_support": "Generic support",
    "support_v3_fixed": "Fixed CleanEdit-SD3",
    "support_v3_controller_rmsgap": "CleanEdit-SD3",
    "flowedit": "FlowEdit-SD3",
    "splitflow": "SplitFlow-SD3",
    "sam_flow_sd3": "Sam-Flow-SD3",
    "fireflow": "FireFlow-FLUX/context",
    "rf_solver_edit": "RF-Solver-Edit-FLUX/context",
    "reflex": "ReFlex-FLUX/context",
    "sam_flow_flux": "Sam-Flow-FLUX/context",
    "dece_rf_flux": "CleanEdit-FLUX/context",
}
METRICS = [
    "outside_l1",
    "outside_mask_l1",
    "bg_psnr",
    "bg_lpips_x100",
    "bg_ssim_luma",
    "source_ssim_luma",
    "dino_source_similarity",
    "bg_dino_source",
    "clip_t",
    "local_clip_t",
    "edit_score",
    "clip_target_minus_source",
    "inside_l1",
    "inside_mask_l1",
    "runtime_seconds",
    "peak_gpu_memory_gb",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if not any(value == key for key, value in row.items())]


def load_sources(paths: list[Path]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    passthrough: list[dict[str, str]] = []
    for path in paths:
        for row in read_csv(path):
            key = (row.get("task", ""), row_method(row), row_seed(row))
            if all(key):
                by_key[key] = row
            else:
                passthrough.append(row)
    return [*passthrough, *by_key.values()]


def row_method(row: dict[str, str]) -> str:
    return row.get("method", "") or row.get("baseline", "")


def row_seed(row: dict[str, str]) -> str:
    return row.get("seed", "").removeprefix("seed_")


def filter_rows(rows: list[dict[str, str]], tasks: list[str], methods: list[str]) -> list[dict[str, str]]:
    task_set = set(tasks)
    method_set = set(methods)
    seed_set = set(SEEDS)
    return [
        row
        for row in rows
        if row.get("task", "") in task_set
        and row_method(row) in method_set
        and row_seed(row) in seed_set
    ]


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


def task_mean_stats(rows: list[dict[str, str]], methods: list[str], tasks: list[str]) -> dict[str, dict[str, float | None]]:
    by_method_task: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method_task[(row_method(row), row.get("task", ""))].append(row)

    out: dict[str, dict[str, float | None]] = {}
    for method in methods:
        metric_task_values: dict[str, list[float]] = defaultdict(list)
        row_count = 0
        success_values = 0
        success_true = 0
        for task in tasks:
            task_rows = by_method_task.get((method, task), [])
            row_count += len(task_rows)
            for metric in METRICS:
                value = avg_values(task_rows, metric)
                if value is not None:
                    metric_task_values[metric].append(value)
            for row in task_rows:
                if row.get("task_success") not in {"", None}:
                    success_values += 1
                    success_true += row.get("task_success") == "True"

        stats: dict[str, float | None] = {"n": float(row_count)}
        for metric in METRICS:
            values = metric_task_values.get(metric, [])
            stats[metric] = mean(values) if len(values) == len(tasks) else None
        stats["task_success_true"] = float(success_true) if success_values else None
        stats["task_success_denominator"] = float(success_values) if success_values else None
        out[method] = stats
    return out


def summary_rows(rows: list[dict[str, str]], methods: list[str], tasks: list[str]) -> list[dict[str, str]]:
    stats = task_mean_stats(rows, methods, tasks)
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
                "outside_l1": fmt(item.get("outside_l1") if item.get("outside_l1") is not None else item.get("outside_mask_l1")),
                "bg_psnr": fmt(item.get("bg_psnr")),
                "bg_lpips_x100": fmt(item.get("bg_lpips_x100")),
                "bg_ssim_luma": fmt(item.get("bg_ssim_luma") if item.get("bg_ssim_luma") is not None else item.get("source_ssim_luma")),
                "source_ssim_luma": fmt(item.get("source_ssim_luma")),
                "dino_source": fmt(item.get("dino_source_similarity")),
                "bg_dino_source": fmt(item.get("bg_dino_source")),
                "clip_t": fmt(item.get("clip_t")),
                "local_clip_t": fmt(item.get("local_clip_t")),
                "edit_score": fmt(item.get("edit_score")),
                "clip_delta": fmt(item.get("clip_target_minus_source")),
                "inside_l1": fmt(item.get("inside_l1") if item.get("inside_l1") is not None else item.get("inside_mask_l1")),
                "runtime_s": fmt(item.get("runtime_seconds")),
                "peak_mem_gb": fmt(item.get("peak_gpu_memory_gb")),
            }
        )
    return output


def family_rows(rows: list[dict[str, str]], methods: list[str]) -> list[dict[str, str]]:
    output = []
    for family, tasks in PHASE2_FAMILIES.items():
        for row in summary_rows(rows, methods, tasks):
            output.append({"family": family, **row})
    return output


def audit_collection(name: str, rows: list[dict[str, str]], tasks: list[str], methods: list[str], metrics: list[str]) -> dict[str, object]:
    expected = {(task, method, seed) for task in tasks for method in methods for seed in SEEDS}
    found = {(row.get("task", ""), row_method(row), row_seed(row)) for row in rows}
    missing = sorted(expected - found)
    duplicates = len(rows) - len(found)
    missing_metrics = {
        metric: sum(1 for row in rows if as_float(row, metric) is None)
        for metric in metrics
    }
    return {
        "name": name,
        "status": "complete" if not missing and duplicates == 0 and not any(missing_metrics.values()) else "incomplete",
        "expected_rows": len(expected),
        "found_rows": len(rows),
        "missing_rows": missing,
        "duplicate_count": duplicates,
        "missing_metrics": missing_metrics,
    }


def count_missing_metric_for_methods(rows: list[dict[str, str]], methods: list[str], metric: str) -> int:
    method_set = set(methods)
    return sum(1 for row in rows if row_method(row) in method_set and as_float(row, metric) is None)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    return md_table_with_headers(rows, fields, fields)


def md_table_with_headers(rows: list[dict[str, str]], fields: list[str], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines)


def write_task_map() -> None:
    rows = []
    for family, tasks in PHASE2_FAMILIES.items():
        for index, task in enumerate(tasks, start=1):
            rows.append({"family": family, "family_index": str(index), "task": task})
    write_csv(OUT_TASK_MAP, rows)


def main() -> int:
    table1_rows = filter_rows(
        load_sources(
            [
                EXP / "e4_t1_t4_reconstruction_floor_metrics.csv",
                EXP / "e1_t1_t4_directgeneric_metrics.csv",
                EXP / "table2a_e4_common_subset_clipdino_metrics.csv",
                EXP / "table2_t5_internal_metrics.csv",
                INTERNAL_BG_METRICS,
            ]
        ),
        PHASE2_TASKS,
        TABLE1_METHODS,
    )
    table2a_rows = filter_rows(
        load_sources(
            [
                EXP / "e1_t1_t4_directgeneric_metrics.csv",
                EXP / "e2_t1_t4_baseline_fixed_mask_metrics.csv",
                EXP / "table2a_e4_common_subset_clipdino_metrics.csv",
                EXP / "table2_t5_internal_metrics.csv",
                EXP / "table2_t5_baseline_metrics.csv",
                INTERNAL_BG_METRICS,
            ]
        ),
        PHASE2_TASKS,
        TABLE2A_METHODS,
    )
    table2b_rows = filter_rows(
        load_sources([EXP / "e2_t1_t4_baseline_fixed_mask_metrics.csv", EXP / "table2_t5_baseline_metrics.csv", EXP / "dece_rf_flux_native_context_metrics.csv"]),
        TABLE2B_TASKS,
        TABLE2B_METHODS,
    )

    audits = [
        audit_collection("table1_phase2_t1_t5_main", table1_rows, PHASE2_TASKS, TABLE1_METHODS, ["outside_l1", "bg_psnr", "bg_lpips_x100", "bg_ssim_luma", "clip_t", "local_clip_t"]),
        audit_collection("table2a_phase2_sd3_common_subset", table2a_rows, PHASE2_TASKS, TABLE2A_METHODS, ["outside_l1", "bg_psnr", "bg_lpips_x100", "bg_ssim_luma", "clip_t", "local_clip_t", "edit_score"]),
        audit_collection("table2b_phase2_native_context", table2b_rows, TABLE2B_TASKS, TABLE2B_METHODS, ["outside_l1", "bg_psnr", "bg_lpips_x100", "bg_ssim_luma", "clip_t", "local_clip_t", "edit_score"]),
    ]
    table1_edit_missing = count_missing_metric_for_methods(
        table1_rows,
        ["direct_target", "adaptive_full_generic_support", "support_v3_controller_rmsgap"],
        "edit_score",
    )
    audits[0]["non_floor_edit_score_missing"] = table1_edit_missing
    if table1_edit_missing:
        audits[0]["status"] = "incomplete"
    complete = all(item["status"] == "complete" for item in audits)

    table1 = summary_rows(table1_rows, TABLE1_METHODS, PHASE2_TASKS)
    table2a = summary_rows(table2a_rows, TABLE2A_METHODS, PHASE2_TASKS)
    table2b = summary_rows(table2b_rows, TABLE2B_METHODS, TABLE2B_TASKS)
    family = family_rows(table1_rows, TABLE1_METHODS)

    write_task_map()
    write_csv(OUT_TABLE1, table1)
    write_csv(OUT_TABLE2A, table2a)
    write_csv(OUT_TABLE2B, table2b)
    write_csv(OUT_FAMILY, family)

    audit_doc = {
        "status": "complete" if complete else "incomplete",
        "scope": "Phase2 locked current project: T1-T5, three source cases per family, seeds 10/11/12.",
        "families": PHASE2_FAMILIES,
        "seeds": SEEDS,
        "audits": audits,
    }
    OUT_AUDIT.write_text(json.dumps(audit_doc, indent=2) + "\n", encoding="utf-8")

    fields = ["label", "n", "outside_l1", "bg_psnr", "bg_lpips_x100", "bg_ssim_luma", "dino_source", "clip_t", "local_clip_t", "edit_score"]
    headers = ["label", "n", "Non-edit MAE ↓", "BG-PSNR ↑", "BG-LPIPS×100 ↓", "BG-SSIM-luma ↑", "DINO-source ↑", "CLIP-T ↑", "Local CLIP-T ↑", "edit_score ↑"]
    supplement_fields = [*fields, "inside_l1", "bg_dino_source", "source_ssim_luma", "clip_delta"]
    supplement_headers = [*headers, "inside_l1 descriptive", "BG-DINO ↑", "Source SSIM-luma ↑", "CLIP delta ↑"]
    lines = [
        "# Phase2 Paper Tables (2026-06-11)",
        "",
        "Current project lock: Phase2 means T1-T5 with three source cases per family. It is the main experiment stage, not a separate task category.",
        "",
        f"Audit status: `{audit_doc['status']}` (`{OUT_AUDIT.relative_to(ROOT)}`).",
        "",
        "## Table 1: Phase2 T1-T5 Main Effect",
        "",
        "Scope: 15 task cases x seeds 10/11/12 = 45 rows per method.",
        "Metrics: Non-edit-region mean absolute error (Non-edit MAE) measures the average pixel deviation between the edited output and source image over the fixed non-edit evaluation region. BG metrics use M_ne = 1 - Dilate(M_edit) to exclude the edit boundary. BG-LPIPS is reported as x100. CLIP-T is full-image target prompt similarity; Local CLIP-T is edit-crop similarity to a task-specific local target phrase. Inside L1 is descriptive and reported only in the supplement block.",
        "",
        r"Formula: NonEdit-MAE = |M_ne|^{-1} || M_ne \odot (\hat{x} - x_src) ||_1.",
        "",
        md_table_with_headers(table1, fields, headers),
        "",
        "## Table 2a: Same-Backbone SD3 Common Subset",
        "",
        md_table_with_headers(table2a, fields, headers),
        "",
        "## Table 2b: Native Preservation-Aware RF / FLUX Context",
        "",
        "Scope: 14 task cases x seeds 10/11/12 = 42 rows per method. The white_bowl_orange_tabletop case is excluded from this native-FLUX comparison (all methods) because the small free-standing orange is below the FLUX generation floor for every native-FLUX method; Tables 1 and 2a retain all 15 cases.",
        "",
        md_table_with_headers(table2b, fields, headers),
        "",
        "## Supplement: Phase2 Family Breakdown",
        "",
        md_table_with_headers(family, ["family", *supplement_fields], ["family", *supplement_headers]),
        "",
        "Do not report the old five-canonical-case Core-5 tables as current evidence. E5 removal remains a separate boundary probe.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"status={audit_doc['status']}")
    for item in audits:
        print(f"{item['name']}: expected={item['expected_rows']} found={item['found_rows']} status={item['status']}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
