from __future__ import annotations

import csv
import html
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
OUT = EXP / "blind_internal_audit_phase2_t1_t5_2026-06-11"

SOURCE_FILES = [
    EXP / "e4_t1_t4_reconstruction_floor_metrics.csv",
    EXP / "e1_t1_t4_directgeneric_metrics.csv",
    EXP / "table2a_e4_common_subset_clipdino_metrics.csv",
    EXP / "table2_t5_internal_metrics.csv",
]
FINAL_REGISTRY = EXP / "phase2_final_selected_runs_2026-06-11.csv"

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

TASK_FAMILY = {
    task: family for family, tasks in PHASE2_FAMILIES.items() for task in tasks
}
TASK_ORDER = [task for tasks in PHASE2_FAMILIES.values() for task in tasks]
METHODS = [
    "base_only",
    "direct_target",
    "adaptive_full_generic_support",
    "support_v3_controller_rmsgap",
]
SEEDS = ["10", "11", "12"]
RATERS = ["rater_01", "rater_02", "rater_03"]
RANDOM_SEED = 20260611
FAILURE_TYPES = [
    "none",
    "semantic_miss",
    "relation_error",
    "under_edit",
    "over_edit",
    "identity_drift",
    "background_drift",
    "locality_leak",
    "artifact",
    "texture_failure",
    "recolor_failure",
    "ambiguous",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if not any(v == k for k, v in r.items())]


def relpath(path: Path) -> str:
    return path.as_posix()


def copy_image(src_value: str, dst: Path) -> bool:
    src = Path(src_value)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def image_suffix(path_value: str) -> str:
    suffix = Path(path_value).suffix.lower()
    return suffix if suffix else ".png"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_html(rater: str, rows: list[dict[str, str]], path: Path) -> None:
    by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_item[row["audit_item_id"]].append(row)

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(rater)} Phase2 blind internal audit</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;line-height:1.35;margin:24px;background:#f7f7f5;color:#202020}",
        ".item{background:white;border:1px solid #ccc;margin:20px 0;padding:16px}",
        ".meta{font-size:14px;color:#333;margin-bottom:10px}",
        ".source{display:flex;gap:16px;align-items:flex-start;margin:10px 0 16px}",
        ".source img{max-width:260px;max-height:220px;border:1px solid #bbb;background:#eee}",
        ".grid{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px}",
        ".card{border:1px solid #bbb;background:#fafafa;padding:10px}",
        ".card img{width:100%;max-height:280px;object-fit:contain;background:#eee;border:1px solid #ddd}",
        ".label{font-weight:bold;margin-bottom:6px}",
        ".rubric{font-size:13px;background:#fffbe8;border:1px solid #d8c77c;padding:10px;margin-bottom:16px}",
        "</style></head><body>",
        f"<h1>{html.escape(rater)} Phase2 Blind Internal Audit</h1>",
        "<div class='rubric'>Internal visual audit only, not a user study. Phase2 is locked as T1-T5 with three source cases per family. Method names are hidden and method order is randomized per item. Fill the matching CSV with 1-5 ratings for edit_correct, relation_correct, source_preservation, locality, artifact_severity, and overall. For artifact_severity, 1 is best and 5 is worst.</div>",
    ]

    for item_id in sorted(by_item):
        item_rows = by_item[item_id]
        first = item_rows[0]
        parts.extend(
            [
                "<div class='item'>",
                f"<h2>{html.escape(item_id)}</h2>",
                f"<div class='meta'><b>Family:</b> {html.escape(first['family'])} &nbsp; <b>Task:</b> {html.escape(first['task'])} &nbsp; <b>Seed:</b> {html.escape(first['seed'])}</div>",
                f"<div class='meta'><b>Target instruction:</b> {html.escape(first['target_prompt'])}</div>",
                "<div class='source'>",
                f"<div><div class='label'>Source</div><img src='{html.escape(first['source_image'])}'></div>",
                f"<div><b>CSV rows:</b> {html.escape(', '.join(r['audit_row_id'] for r in item_rows))}<br>Rate each anonymous candidate independently.</div>",
                "</div>",
                "<div class='grid'>",
            ]
        )
        for row in item_rows:
            parts.extend(
                [
                    "<div class='card'>",
                    f"<div class='label'>Candidate {html.escape(row['candidate_label'])}</div>",
                    f"<img src='{html.escape(row['candidate_image'])}'>",
                    f"<div class='meta'>row: {html.escape(row['audit_row_id'])}</div>",
                    "</div>",
                ]
            )
        parts.extend(["</div>", "</div>"])

    parts.extend(["</body></html>"])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def load_metrics() -> dict[tuple[str, str, str], dict[str, str]]:
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in SOURCE_FILES:
        for row in read_rows(path):
            task = row.get("task", "")
            method = row.get("method", "")
            seed = row.get("seed", "").removeprefix("seed_")
            if task not in TASK_FAMILY or method not in METHODS or seed not in SEEDS:
                continue
            key = (task, seed, method)
            if key in by_key:
                raise RuntimeError(f"duplicate metric row for {key} in {path}")
            by_key[key] = row
    return by_key


def load_registry() -> dict[tuple[str, str, str], dict[str, str]]:
    if not FINAL_REGISTRY.exists():
        return {}
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_rows(FINAL_REGISTRY):
        task = row.get("task", "")
        method = row.get("method", "")
        seed = row.get("seed", "").removeprefix("seed_")
        if task in TASK_FAMILY and method in METHODS and seed in SEEDS:
            by_key[(task, seed, method)] = row
    return by_key


def registry_value(
    registry: dict[tuple[str, str, str], dict[str, str]],
    task: str,
    seed: str,
    method: str,
    field: str,
) -> str:
    return (registry.get((task, seed, method), {}).get(field) or "").strip()


def registry_status_counts(registry: dict[tuple[str, str, str], dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in registry.values():
        status = row.get("metric_status", "") or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    by_key = load_metrics()
    registry = load_registry()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    private_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    rater_row_counts: dict[str, int] = {}

    for rater in RATERS:
        rater_dir = OUT / rater
        image_dir = rater_dir / "images"
        sheet_rows: list[dict[str, str]] = []
        item_number = 0
        for task in TASK_ORDER:
            family = TASK_FAMILY[task]
            for seed in SEEDS:
                item_number += 1
                item_id = f"item_{item_number:03d}"
                method_order = METHODS[:]
                rng.shuffle(method_order)

                source_row = next(
                    (by_key.get((task, seed, method)) for method in METHODS if by_key.get((task, seed, method))),
                    None,
                )
                if not source_row:
                    missing.append({"rater": rater, "family": family, "task": task, "seed": seed, "method": "", "kind": "source_row"})
                    continue
                source_image = (
                    registry_value(registry, task, seed, "base_only", "selected_source_image")
                    or source_row.get("source_image", "")
                )
                source_suffix = image_suffix(source_image)
                source_dst = image_dir / f"{item_id}_source{source_suffix}"
                if not copy_image(source_image, source_dst):
                    missing.append({"rater": rater, "family": family, "task": task, "seed": seed, "method": "", "kind": "source_image"})
                source_rel = relpath(source_dst.relative_to(rater_dir))

                for label_index, method in enumerate(method_order):
                    label = chr(ord("A") + label_index)
                    row = by_key.get((task, seed, method))
                    if not row:
                        missing.append({"rater": rater, "family": family, "task": task, "seed": seed, "method": method, "kind": "result_row"})
                        continue

                    selected_image = registry_value(registry, task, seed, method, "selected_result_image") or row.get("result_image", "")
                    selected_run = registry_value(registry, task, seed, method, "selected_run") or method
                    selection_reason = registry_value(registry, task, seed, method, "selection_reason")
                    metric_status = registry_value(registry, task, seed, method, "metric_status") or "no_registry"
                    result_suffix = image_suffix(selected_image)
                    anon_name = f"{item_id}_{label}{result_suffix}"
                    result_dst = image_dir / anon_name
                    if not copy_image(selected_image, result_dst):
                        missing.append({"rater": rater, "family": family, "task": task, "seed": seed, "method": method, "kind": "result_image"})
                    candidate_rel = relpath(result_dst.relative_to(rater_dir))
                    audit_row_id = f"{rater}_{item_id}_{label}"

                    public = {
                        "rater_id": rater,
                        "audit_item_id": item_id,
                        "audit_row_id": audit_row_id,
                        "family": family,
                        "task": task,
                        "seed": seed,
                        "candidate_label": label,
                        "source_image": source_rel,
                        "candidate_image": candidate_rel,
                        "target_prompt": row.get("target_prompt", ""),
                        "edit_correct_1_5": "",
                        "relation_correct_1_5": "",
                        "source_preservation_1_5": "",
                        "locality_1_5": "",
                        "artifact_severity_1_5": "",
                        "overall_1_5": "",
                        "failure_type": "",
                        "notes": "",
                    }
                    sheet_rows.append(public)
                    manifest_rows.append(public.copy())
                    private_rows.append(
                        {
                            "rater_id": rater,
                            "audit_item_id": item_id,
                            "audit_row_id": audit_row_id,
                            "family": family,
                            "task": task,
                            "seed": seed,
                            "candidate_label": label,
                            "method": method,
                            "method_display": row.get("method_display", ""),
                            "source_image_original": source_image,
                            "result_image_original": selected_image,
                            "metric_result_image_original": row.get("result_image", ""),
                            "selected_run": selected_run,
                            "selection_reason": selection_reason,
                            "metric_status": metric_status,
                            "anonymous_image": f"{rater}/{candidate_rel}",
                        }
                    )

        write_csv(rater_dir / f"{rater}_sheet.csv", sheet_rows)
        build_html(rater, sheet_rows, rater_dir / f"{rater}_index.html")
        rater_row_counts[rater] = len(sheet_rows)

    write_csv(OUT / "blind_internal_audit_manifest.csv", manifest_rows)
    write_csv(OUT / "blind_internal_audit_method_key_private.csv", private_rows)
    if missing:
        write_csv(OUT / "blind_internal_audit_missing.csv", missing)

    readme = [
        "# Blind Internal Audit Package (Phase2 T1-T5)",
        "",
        "Internal visual audit only; do not call this a user study.",
        "",
        "Current project lock: Phase2 means T1-T5 with three source cases per family. This package covers 15 task cases x seeds 10/11/12 x 4 paper-facing methods.",
        "",
        "Image selection follows `phase2_final_selected_runs_2026-06-11.csv` when available. The private method key records the selected run, selection reason, and metric-alignment status for each anonymous image.",
        "",
        "Rater workflow:",
        "",
        "1. Open `rater_XX/rater_XX_index.html`.",
        "2. Fill the matching `rater_XX/rater_XX_sheet.csv`.",
        "3. Score each anonymous candidate independently.",
        "",
        "Rubric:",
        "",
        "- `edit_correct_1_5`: target edit is present and visually convincing.",
        "- `relation_correct_1_5`: edit is in the requested relation/location.",
        "- `source_preservation_1_5`: source identity, layout, and background are preserved.",
        "- `locality_1_5`: change is localized to the intended edit region.",
        "- `artifact_severity_1_5`: 1 is clean/best, 5 is severe/worst.",
        "- `overall_1_5`: overall usefulness for the requested local edit.",
        "- `failure_type`: one of " + ", ".join(FAILURE_TYPES) + ".",
        "",
        "Method names are hidden in rater files. Keep `blind_internal_audit_method_key_private.csv` private until all rater sheets are frozen.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    expected_items_per_rater = len(TASK_ORDER) * len(SEEDS)
    expected_rows_per_rater = expected_items_per_rater * len(METHODS)
    expected_rows_total = expected_rows_per_rater * len(RATERS)
    complete = (
        not missing
        and len(manifest_rows) == expected_rows_total
        and len(private_rows) == expected_rows_total
        and all(count == expected_rows_per_rater for count in rater_row_counts.values())
    )
    audit = {
        "status": "complete" if complete else "incomplete",
        "scope": "Phase2 locked current project: T1-T5, three source cases per family, seeds 10/11/12.",
        "random_seed": RANDOM_SEED,
        "raters": RATERS,
        "families": PHASE2_FAMILIES,
        "seeds": SEEDS,
        "methods": METHODS,
        "methods_per_item": len(METHODS),
        "expected_items_per_rater": expected_items_per_rater,
        "expected_rows_per_rater": expected_rows_per_rater,
        "expected_manifest_rows": expected_rows_total,
        "rater_row_counts": rater_row_counts,
        "manifest_rows": len(manifest_rows),
        "private_key_rows": len(private_rows),
        "missing_count": len(missing),
        "method_names_hidden_in_rater_files": True,
        "final_selected_run_registry": str(FINAL_REGISTRY.relative_to(ROOT)) if FINAL_REGISTRY.exists() else "",
        "registry_rows_loaded": len(registry),
        "registry_metric_status_counts": registry_status_counts(registry),
    }
    (OUT / "blind_internal_audit_package_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"status={audit['status']}")
    print(f"manifest_rows={audit['manifest_rows']} private_key_rows={audit['private_key_rows']} missing={audit['missing_count']}")
    print(f"wrote {OUT}")
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
