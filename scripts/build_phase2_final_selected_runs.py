from __future__ import annotations

import csv
import json
from pathlib import Path


REMOTE_PROJECT = Path("/cluster/users/grad/2025/25t8103/project")
if REMOTE_PROJECT.exists():
    PROJECT = REMOTE_PROJECT
    EXP = PROJECT / "experiments" / "support_v3_2026-06-02"
else:
    PROJECT = Path(r"I:\Downloads\1\2")
    EXP = PROJECT / "phase2_lock_2026-06-11"

OUT_CSV = EXP / "phase2_final_selected_runs_2026-06-11.csv"
OUT_JSON = EXP / "phase2_final_selected_runs_2026-06-11.json"

SEEDS = ["10", "11", "12"]

FAMILIES = [
    (
        "T1_attached_accessory",
        [
            ("cat_crown", "Cat crown"),
            ("dog_bow_tie_phase2", "Dog bow tie"),
            ("dog_front_sunglasses_phase2", "Dog sunglasses"),
        ],
    ),
    (
        "T2_container_insertion",
        [
            ("bowl_apple_inside", "Bowl + apple"),
            ("white_bowl_orange_tabletop_phase2", "White bowl + orange"),
            ("brown_bowl_lemon_phase2", "Brown bowl + lemon"),
        ],
    ),
    (
        "T3_surface_decal",
        [
            ("tshirt_star", "T-shirt star"),
            ("mug_heart", "Mug heart"),
            ("tote_leaf", "Tote leaf"),
        ],
    ),
    (
        "T4_local_recolor",
        [
            ("red_office_chair_to_blue_office_chair", "Office chair blue"),
            ("green_mug_orange_phase2", "Green mug orange"),
            ("yellow_vase_blue_phase2", "Yellow vase blue"),
        ],
    ),
    (
        "T5_same_color_material",
        [
            ("pillow_same_color_cable_knit", "White cable-knit pillow"),
            ("pillow_same_color_cable_knit_grey", "Grey cable-knit pillow"),
            ("pillow_same_color_cable_knit_armchair", "Armchair cable-knit pillow"),
        ],
    ),
]

T1_T4_TASKS = {task for family, tasks in FAMILIES[:4] for task, _ in tasks}
T5_TASKS = {task for _, tasks in FAMILIES[4:] for task, _ in tasks}

METHODS = [
    ("base_only", "RF recon"),
    ("direct_target", "Direct target"),
    ("adaptive_full_generic_support", "Generic support"),
    ("flowedit", "FlowEdit"),
    ("splitflow", "SplitFlow"),
    ("sam_flow_sd3", "Sam-Flow-SD3"),
    ("fireflow", "FireFlow"),
    ("rf_solver_edit", "RF-Solver-Edit"),
    ("reflex", "ReFlex"),
    ("sam_flow_flux", "Sam-Flow-FLUX"),
    ("support_v3_fixed", "Fixed DeCE"),
    ("support_v3_controller_rmsgap", "DeCE-RF"),
]

METRIC_FILES = [
    EXP / "phase2_internal_bg_metrics.csv",
    EXP / "strict_fixed_mask_metrics.csv",
    EXP / "e1_t1_t4_directgeneric_metrics.csv",
    EXP / "table2a_e4_common_subset_clipdino_metrics.csv",
    EXP / "e2_t1_t4_baseline_fixed_mask_metrics.csv",
    EXP / "table2_t5_internal_metrics.csv",
    EXP / "table2_t5_baseline_metrics.csv",
    EXP / "e4_t1_t4_reconstruction_floor_metrics.csv",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def norm_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def load_metric_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in METRIC_FILES:
        for row in read_csv(path):
            task = row.get("task", "")
            seed = row.get("seed", "").removeprefix("seed_")
            method = row.get("method", "")
            if task and seed and method:
                item = dict(row)
                item["_metric_file"] = norm_path(path)
                rows.setdefault((task, seed, method), item)
    return rows


def load_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in [
        PROJECT / "data" / "phase2_candidates" / "sources_manifest.csv",
        EXP / "normalized_512" / "t1_t4_eval_assets_manifest.csv",
        EXP / "normalized_512" / "t5_eval_assets_manifest.csv",
    ]:
        for row in read_csv(path):
            task = row.get("task", "")
            image = row.get("source_image", "") or row.get("source", "")
            if task and image:
                sources[task] = image
    return sources


def source_image(task: str, sources: dict[str, str]) -> str:
    normalized = EXP / "normalized_512" / "sources" / f"{task}.png"
    if normalized.exists():
        return norm_path(normalized)
    return sources.get(task, "")


def selected_run(task: str, method: str) -> tuple[str, str]:
    if method == "support_v3_controller_rmsgap":
        if task == "white_bowl_orange_tabletop_phase2":
            return (
                "support_v3_controller_rmsgap_failed3_fix6_20260609",
                "selected repaired DeCE-RF run with visible orange; plain seed 12 is a weak earlier run",
            )
        if task == "tshirt_star":
            return (
                "support_v3_controller_rmsgap",
                "selected natural-looking DeCE-RF run: visible red star follows shirt lighting/folds; t1t4 rerun lost the star and fix3 looks overlaid",
            )
        if task == "mug_heart":
            return (
                "support_v3_controller_rmsgap_mugbox145_c180_ref075_v1",
                "selected unified mug_heart c180 run with deterministic final-ref-composite scale 0.75 for seeds 10/11/12",
            )
        if task in T1_T4_TASKS:
            return (
                "support_v3_controller_rmsgap_t1t4_3seed_20260609",
                "formal T1-T4 three-seed DeCE-RF rerun",
            )
        return ("support_v3_controller_rmsgap", "formal T5 DeCE-RF run")
    return (method, "canonical method directory")


def result_path(task: str, seed: str, method: str, run: str, metric_row: dict[str, str] | None) -> Path | None:
    candidates: list[Path] = []
    if method in {
        "flowedit",
        "splitflow",
        "sam_flow_sd3",
        "fireflow",
        "rf_solver_edit",
        "reflex",
        "sam_flow_flux",
    }:
        candidates.extend(
            [
                PROJECT / "outputs" / "e2_t1_t4_baseline_matrix" / task / method / f"seed_{seed}" / "result.png",
                PROJECT / "outputs" / "e2_t5_baseline_matrix" / task / method / f"seed_{seed}" / "result.png",
                PROJECT / "outputs" / "baselines" / method / task / f"seed_{seed}" / "result.png",
            ]
        )
    else:
        candidates.append(PROJECT / "outputs" / "pretty_matrix" / task / run / f"seed_{seed}" / "result.png")
    if metric_row and metric_row.get("result_image"):
        candidates.append(PROJECT / metric_row["result_image"])
        candidates.append(Path(metric_row["result_image"]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def metric_status(selected: Path | None, metric_row: dict[str, str] | None) -> tuple[str, str]:
    if not selected:
        return "missing_selected_image", "selected result image is missing"
    if not metric_row:
        return "no_metric_row", "no matching metric row was found"
    metric_image = metric_row.get("result_image", "")
    if not metric_image:
        return "no_metric_image", "metric row has no result_image"
    selected_abs = selected.resolve()
    metric_candidates = [PROJECT / metric_image, Path(metric_image)]
    for candidate in metric_candidates:
        try:
            if candidate.exists() and candidate.resolve() == selected_abs:
                return "aligned", "selected image matches metric row result_image"
        except OSError:
            pass
    return "needs_metric_refresh", "selected visual run differs from current metric row result_image"


def build_rows() -> list[dict[str, str]]:
    metric_rows = load_metric_rows()
    sources = load_sources()
    rows: list[dict[str, str]] = []
    label_by_task = {task: label for _, tasks in FAMILIES for task, label in tasks}
    family_by_task = {task: family for family, tasks in FAMILIES for task, _ in tasks}

    for family, tasks in FAMILIES:
        for task, task_label in tasks:
            for seed in SEEDS:
                for method, method_display in METHODS:
                    run, reason = selected_run(task, method)
                    metric_row = metric_rows.get((task, seed, method))
                    selected = result_path(task, seed, method, run, metric_row)
                    status, note = metric_status(selected, metric_row)
                    rows.append(
                        {
                            "family": family_by_task[task],
                            "task": task,
                            "task_label": label_by_task[task],
                            "seed": seed,
                            "method": method,
                            "method_display": method_display,
                            "selected_run": run,
                            "selected_result_image": norm_path(selected) if selected else "",
                            "selected_source_image": source_image(task, sources),
                            "selection_reason": reason,
                            "metric_status": status,
                            "metric_note": note,
                            "metric_file": metric_row.get("_metric_file", "") if metric_row else "",
                            "metric_result_image": metric_row.get("result_image", "") if metric_row else "",
                        }
                    )
    return rows


def main() -> None:
    rows = build_rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "task",
        "task_label",
        "seed",
        "method",
        "method_display",
        "selected_run",
        "selected_result_image",
        "selected_source_image",
        "selection_reason",
        "metric_status",
        "metric_note",
        "metric_file",
        "metric_result_image",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["metric_status"]] = counts.get(row["metric_status"], 0) + 1
    audit = {
        "status": "complete" if not counts.get("missing_selected_image") else "incomplete",
        "scope": "Phase2 final selected run registry for visual artifacts; seeds 10/11/12; T1-T5.",
        "csv": norm_path(OUT_CSV),
        "rows": len(rows),
        "metric_status_counts": counts,
        "selection_rules": {
            "white_bowl_orange_tabletop_phase2/support_v3_controller_rmsgap": "support_v3_controller_rmsgap_failed3_fix6_20260609",
            "tshirt_star/support_v3_controller_rmsgap": "support_v3_controller_rmsgap",
            "mug_heart/support_v3_controller_rmsgap": "support_v3_controller_rmsgap_mugbox145_c180_ref075_v1",
            "T1-T4/support_v3_controller_rmsgap": "support_v3_controller_rmsgap_t1t4_3seed_20260609 except explicit repaired visual overrides",
            "T5/support_v3_controller_rmsgap": "support_v3_controller_rmsgap",
            "other methods": "canonical method or baseline matrix directory",
        },
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"status={audit['status']}")
    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print(f"metric_status_counts={counts}")
    if audit["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
