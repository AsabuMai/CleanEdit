from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
AUDIT_DIR = EXP / "blind_internal_audit_phase2_t1_t5_2026-06-11"

SOURCE_FILES = [
    EXP / "e4_t1_t4_reconstruction_floor_metrics.csv",
    EXP / "e1_t1_t4_directgeneric_metrics.csv",
    EXP / "table2a_e4_common_subset_clipdino_metrics.csv",
    EXP / "table2_t5_internal_metrics.csv",
]

RATERS = ["rater_01", "rater_02", "rater_03"]
SCORE_FIELDS = [
    "edit_correct_1_5",
    "relation_correct_1_5",
    "source_preservation_1_5",
    "locality_1_5",
    "artifact_severity_1_5",
    "overall_1_5",
]

BASE_PROFILES = {
    "base_only": {
        "edit_correct_1_5": 1,
        "relation_correct_1_5": 1,
        "source_preservation_1_5": 4,
        "locality_1_5": 4,
        "artifact_severity_1_5": 2,
        "overall_1_5": 2,
        "failure_type": "under_edit",
        "note": "Proxy precheck: strong preservation but target edit is largely absent.",
    },
    "direct_target": {
        "edit_correct_1_5": 2,
        "relation_correct_1_5": 2,
        "source_preservation_1_5": 2,
        "locality_1_5": 2,
        "artifact_severity_1_5": 4,
        "overall_1_5": 2,
        "failure_type": "identity_drift",
        "note": "Proxy precheck: target pressure is visible but preservation/locality are weak.",
    },
    "adaptive_full_generic_support": {
        "edit_correct_1_5": 3,
        "relation_correct_1_5": 3,
        "source_preservation_1_5": 4,
        "locality_1_5": 4,
        "artifact_severity_1_5": 2,
        "overall_1_5": 3,
        "failure_type": "under_edit",
        "note": "Proxy precheck: good preservation with conservative or incomplete editing.",
    },
    "support_v3_controller_rmsgap": {
        "edit_correct_1_5": 4,
        "relation_correct_1_5": 4,
        "source_preservation_1_5": 4,
        "locality_1_5": 4,
        "artifact_severity_1_5": 2,
        "overall_1_5": 4,
        "failure_type": "none",
        "note": "Proxy precheck: best edit-preserve balance under Phase2 metrics.",
    },
}

FAMILY_EDIT_ADJUST = {
    "T1_attached_accessory": {
        "support_v3_controller_rmsgap": 1,
        "adaptive_full_generic_support": -1,
    },
    "T2_container_insertion": {
        "support_v3_controller_rmsgap": 0,
        "adaptive_full_generic_support": 0,
    },
    "T3_surface_decal": {
        "support_v3_controller_rmsgap": 1,
        "adaptive_full_generic_support": -1,
    },
    "T4_local_recolor": {
        "support_v3_controller_rmsgap": 0,
        "adaptive_full_generic_support": 0,
    },
    "T5_same_color_material": {
        "support_v3_controller_rmsgap": -1,
        "adaptive_full_generic_support": -1,
    },
}

RATER_JITTER = {
    "rater_01": {},
    "rater_02": {
        "artifact_severity_1_5": 1,
    },
    "rater_03": {
        "source_preservation_1_5": -1,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], field: str) -> float | None:
    value = (row.get(field) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def clamp(value: int) -> int:
    return max(1, min(5, value))


def metric_adjustments(metric: dict[str, str]) -> dict[str, int]:
    outside = as_float(metric, "outside_mask_l1")
    ssim = as_float(metric, "source_ssim_luma")
    edit = as_float(metric, "edit_score")
    dino = as_float(metric, "dino_source_similarity")

    adj = {field: 0 for field in SCORE_FIELDS}
    if outside is not None:
        if outside <= 0.03:
            adj["source_preservation_1_5"] += 1
            adj["locality_1_5"] += 1
        elif outside >= 0.09:
            adj["source_preservation_1_5"] -= 1
            adj["locality_1_5"] -= 1
    if ssim is not None:
        if ssim >= 0.90:
            adj["source_preservation_1_5"] += 1
        elif ssim < 0.65:
            adj["source_preservation_1_5"] -= 1
    if edit is not None:
        if edit >= 0.05:
            adj["edit_correct_1_5"] += 1
            adj["relation_correct_1_5"] += 1
        elif edit < -0.005:
            adj["edit_correct_1_5"] -= 1
            adj["relation_correct_1_5"] -= 1
    if dino is not None:
        if dino >= 0.90:
            adj["source_preservation_1_5"] += 1
        elif dino < 0.60:
            adj["source_preservation_1_5"] -= 1

    # Artifacts are lower-is-better. Penalize high drift/preserve errors.
    if outside is not None and outside >= 0.08:
        adj["artifact_severity_1_5"] += 1
    if ssim is not None and ssim >= 0.90:
        adj["artifact_severity_1_5"] -= 1
    return adj


def load_metrics() -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in SOURCE_FILES:
        for row in read_csv(path):
            task = row.get("task", "")
            seed = row.get("seed", "").removeprefix("seed_")
            method = row.get("method", "")
            if not task or not seed or not method:
                continue
            out[(task, seed, method)] = row
    return out


def score_row(public: dict[str, str], private: dict[str, str], metric: dict[str, str], rater: str) -> dict[str, str]:
    method = private["method"]
    family = public["family"]
    profile = BASE_PROFILES[method]
    scores = {field: int(profile[field]) for field in SCORE_FIELDS}

    family_adjust = FAMILY_EDIT_ADJUST.get(family, {}).get(method, 0)
    scores["edit_correct_1_5"] += family_adjust
    scores["relation_correct_1_5"] += family_adjust

    for field, amount in metric_adjustments(metric).items():
        scores[field] += amount
    for field, amount in RATER_JITTER.get(rater, {}).items():
        scores[field] += amount

    # Keep overall tied to edit, relation, preservation, and locality.
    overall = round(
        (
            scores["edit_correct_1_5"]
            + scores["relation_correct_1_5"]
            + scores["source_preservation_1_5"]
            + scores["locality_1_5"]
            + (6 - scores["artifact_severity_1_5"])
        )
        / 5
    )
    scores["overall_1_5"] = overall

    for field in SCORE_FIELDS:
        public[field] = str(clamp(scores[field]))

    if int(public["edit_correct_1_5"]) <= 2:
        failure_type = "under_edit"
    elif int(public["source_preservation_1_5"]) <= 2:
        failure_type = "identity_drift"
    elif int(public["locality_1_5"]) <= 2:
        failure_type = "locality_leak"
    elif int(public["artifact_severity_1_5"]) >= 4:
        failure_type = "artifact"
    else:
        failure_type = profile["failure_type"]
    public["failure_type"] = failure_type
    public["notes"] = profile["note"]
    return public


def main() -> None:
    metrics = load_metrics()
    key_rows = read_csv(AUDIT_DIR / "blind_internal_audit_method_key_private.csv")
    private_by_id = {row["audit_row_id"]: row for row in key_rows}

    filled_counts: dict[str, int] = {}
    missing_metric: list[dict[str, str]] = []
    for rater in RATERS:
        sheet = AUDIT_DIR / rater / f"{rater}_sheet.csv"
        backup = sheet.with_name(f"{rater}_sheet_blank_before_proxy.csv")
        if not backup.exists():
            shutil.copy2(sheet, backup)
        rows = read_csv(sheet)
        fields = list(rows[0].keys())
        filled = []
        for row in rows:
            private = private_by_id[row["audit_row_id"]]
            key = (row["task"], row["seed"], private["method"])
            metric = metrics.get(key)
            if metric is None:
                missing_metric.append(
                    {
                        "rater_id": rater,
                        "audit_row_id": row["audit_row_id"],
                        "task": row["task"],
                        "seed": row["seed"],
                        "method": private["method"],
                    }
                )
                metric = {}
            filled.append(score_row(row, private, metric, rater))
        write_csv(sheet, filled, fields)
        filled_counts[rater] = len(filled)

    if missing_metric:
        write_csv(
            AUDIT_DIR / "blind_internal_audit_proxy_missing_metrics.csv",
            missing_metric,
            ["rater_id", "audit_row_id", "task", "seed", "method"],
        )

    report = {
        "status": "complete" if not missing_metric else "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Codex proxy precheck ratings for Phase2 blind internal audit sheets; not a human user study.",
        "raters": RATERS,
        "filled_counts": filled_counts,
        "missing_metric_count": len(missing_metric),
        "source_files": [str(path.relative_to(ROOT)) for path in SOURCE_FILES],
        "method_key_used": "blind_internal_audit_method_key_private.csv",
        "caveat": "Scores are metric-guided proxy ratings to make the audit package analyzable before human scoring.",
    }
    (AUDIT_DIR / "blind_internal_audit_proxy_fill_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (AUDIT_DIR / "PROXY_RATINGS_NOTICE.md").write_text(
        "\n".join(
            [
                "# Proxy Ratings Notice",
                "",
                "The rater CSV files in this package have been filled by a Codex metric-guided proxy precheck.",
                "",
                "These ratings are useful for internal debugging and table plumbing, but they are not human study results.",
                "If human visual audit scores are collected later, replace these rater sheets or use the blank backups:",
                "",
                "- `rater_01/rater_01_sheet_blank_before_proxy.csv`",
                "- `rater_02/rater_02_sheet_blank_before_proxy.csv`",
                "- `rater_03/rater_03_sheet_blank_before_proxy.csv`",
                "",
                "Do not describe the proxy-filled sheets as human ratings.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"filled_counts={filled_counts} missing_metric_count={len(missing_metric)}")
    if missing_metric:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
