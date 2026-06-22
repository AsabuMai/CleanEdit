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
METRICS = EXP / "e1_core5_strict_metrics.csv"
OUT = EXP / "blind_internal_audit_2026-06-11"

TASK_ORDER = [
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
        f"<title>{html.escape(rater)} blind internal audit</title>",
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
        f"<h1>{html.escape(rater)} Blind Internal Audit</h1>",
        "<div class='rubric'>Internal visual audit only, not a user study. Method names are hidden and method order is randomized per item. Fill the matching CSV with 1-5 ratings for edit_correct, relation_correct, source_preservation, locality, artifact_severity, and overall. For artifact_severity, 1 is best and 5 is worst.</div>",
    ]

    for item_id in sorted(by_item):
        item_rows = by_item[item_id]
        first = item_rows[0]
        parts.extend(
            [
                "<div class='item'>",
                f"<h2>{html.escape(item_id)}</h2>",
                f"<div class='meta'><b>Task:</b> {html.escape(first['task'])} &nbsp; <b>Seed:</b> {html.escape(first['seed'])}</div>",
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


def main() -> None:
    rows = read_rows(METRICS)
    by_key = {
        (row.get("task", ""), row.get("seed", "").removeprefix("seed_"), row.get("method", "")): row
        for row in rows
        if row.get("task") in TASK_ORDER and row.get("method") in METHODS
    }

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    private_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for rater_index, rater in enumerate(RATERS, start=1):
        rater_dir = OUT / rater
        image_dir = rater_dir / "images"
        sheet_rows: list[dict[str, str]] = []
        item_number = 0
        for task in TASK_ORDER:
            for seed in SEEDS:
                item_number += 1
                item_id = f"item_{item_number:03d}"
                method_order = METHODS[:]
                rng.shuffle(method_order)

                source_row = by_key.get((task, seed, method_order[0]))
                if not source_row:
                    missing.append({"rater": rater, "task": task, "seed": seed, "method": method_order[0], "kind": "source_row"})
                    continue
                source_suffix = image_suffix(source_row.get("source_image", ""))
                source_dst = image_dir / f"{item_id}_source{source_suffix}"
                if not copy_image(source_row.get("source_image", ""), source_dst):
                    missing.append({"rater": rater, "task": task, "seed": seed, "method": method_order[0], "kind": "source_image"})
                source_rel = relpath(source_dst.relative_to(rater_dir))

                for label_index, method in enumerate(method_order):
                    label = chr(ord("A") + label_index)
                    row = by_key.get((task, seed, method))
                    if not row:
                        missing.append({"rater": rater, "task": task, "seed": seed, "method": method, "kind": "result_row"})
                        continue

                    result_suffix = image_suffix(row.get("result_image", ""))
                    anon_name = f"{item_id}_{label}{result_suffix}"
                    result_dst = image_dir / anon_name
                    if not copy_image(row.get("result_image", ""), result_dst):
                        missing.append({"rater": rater, "task": task, "seed": seed, "method": method, "kind": "result_image"})
                    candidate_rel = relpath(result_dst.relative_to(rater_dir))
                    audit_row_id = f"{rater}_{item_id}_{label}"

                    public = {
                        "rater_id": rater,
                        "audit_item_id": item_id,
                        "audit_row_id": audit_row_id,
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
                            "task": task,
                            "seed": seed,
                            "candidate_label": label,
                            "method": method,
                            "method_display": row.get("method_display", ""),
                            "source_image_original": row.get("source_image", ""),
                            "result_image_original": row.get("result_image", ""),
                            "anonymous_image": f"{rater}/{candidate_rel}",
                        }
                    )

        write_csv(rater_dir / f"{rater}_sheet.csv", sheet_rows)
        build_html(rater, sheet_rows, rater_dir / f"{rater}_index.html")

    write_csv(OUT / "blind_internal_audit_manifest.csv", manifest_rows)
    write_csv(OUT / "blind_internal_audit_method_key_private.csv", private_rows)
    if missing:
        write_csv(OUT / "blind_internal_audit_missing.csv", missing)

    readme = [
        "# Blind Internal Audit Package (Core-5)",
        "",
        "Internal visual audit only; do not call this a user study.",
        "",
        "Scope: strict Core-5 Table 1 outputs, 5 tasks x 3 seeds x 4 paper-facing methods.",
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

    audit = {
        "status": "complete" if not missing else "incomplete",
        "scope": "strict Core-5 Table 1 blind internal audit package",
        "random_seed": RANDOM_SEED,
        "raters": RATERS,
        "tasks": TASK_ORDER,
        "seeds": SEEDS,
        "methods_per_item": len(METHODS),
        "expected_items_per_rater": len(TASK_ORDER) * len(SEEDS),
        "expected_rows_per_rater": len(TASK_ORDER) * len(SEEDS) * len(METHODS),
        "manifest_rows": len(manifest_rows),
        "private_key_rows": len(private_rows),
        "missing_count": len(missing),
        "method_names_hidden_in_rater_files": True,
    }
    (OUT / "blind_internal_audit_package_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"status={audit['status']}")
    print(f"manifest_rows={audit['manifest_rows']} private_key_rows={audit['private_key_rows']} missing={audit['missing_count']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
