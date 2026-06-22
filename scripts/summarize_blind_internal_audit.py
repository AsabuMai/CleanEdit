from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = ROOT / "experiments" / "support_v3_2026-06-02" / "blind_internal_audit_phase2_t1_t5_2026-06-11"

SCORE_FIELDS = [
    "edit_correct_1_5",
    "relation_correct_1_5",
    "source_preservation_1_5",
    "locality_1_5",
    "artifact_severity_1_5",
    "overall_1_5",
]
RATERS = ["rater_01", "rater_02", "rater_03"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_score(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        score = float(value)
    except ValueError:
        return None
    if score < 1 or score > 5:
        return None
    return score


def avg(rows: list[dict[str, str]], field: str) -> str:
    vals = [v for row in rows if (v := to_score(row.get(field, ""))) is not None]
    return f"{mean(vals):.3f}" if vals else ""


def summarize_group(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        item = {k: v for k, v in zip(keys, key)}
        item["n"] = str(len(group))
        for field in SCORE_FIELDS:
            item[field + "_mean"] = avg(group, field)
        failures = Counter((row.get("failure_type") or "").strip() for row in group if (row.get("failure_type") or "").strip())
        item["failure_type_counts"] = ";".join(f"{name}:{count}" for name, count in sorted(failures.items()))
        out.append(item)
    return out


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines)


def main() -> None:
    audit_dir = DEFAULT_AUDIT_DIR
    key_path = audit_dir / "blind_internal_audit_method_key_private.csv"
    key_rows = read_csv(key_path)
    by_id = {row["audit_row_id"]: row for row in key_rows}

    merged: list[dict[str, str]] = []
    missing_scores: list[dict[str, str]] = []
    invalid_scores: list[dict[str, str]] = []
    missing_sheet: list[str] = []

    for rater in RATERS:
        sheet = audit_dir / rater / f"{rater}_sheet.csv"
        if not sheet.exists():
            missing_sheet.append(str(sheet))
            continue
        for row in read_csv(sheet):
            audit_row_id = row.get("audit_row_id", "")
            private = by_id.get(audit_row_id, {})
            merged_row = {
                **row,
                "method": private.get("method", ""),
                "method_display": private.get("method_display", ""),
            }
            merged.append(merged_row)
            for field in SCORE_FIELDS:
                value = row.get(field, "")
                if not (value or "").strip():
                    missing_scores.append({"audit_row_id": audit_row_id, "rater_id": rater, "field": field})
                elif to_score(value) is None:
                    invalid_scores.append({"audit_row_id": audit_row_id, "rater_id": rater, "field": field, "value": value})

    complete = not missing_sheet and not missing_scores and not invalid_scores and len(merged) == len(key_rows)
    audit = {
        "status": "complete" if complete else "incomplete",
        "expected_rows": len(key_rows),
        "merged_rows": len(merged),
        "missing_sheet_count": len(missing_sheet),
        "missing_score_count": len(missing_scores),
        "invalid_score_count": len(invalid_scores),
    }
    (audit_dir / "blind_internal_audit_score_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    if missing_scores:
        write_csv(audit_dir / "blind_internal_audit_missing_scores.csv", missing_scores, ["audit_row_id", "rater_id", "field"])
    if invalid_scores:
        write_csv(audit_dir / "blind_internal_audit_invalid_scores.csv", invalid_scores, ["audit_row_id", "rater_id", "field", "value"])
    if missing_sheet:
        (audit_dir / "blind_internal_audit_missing_sheets.txt").write_text("\n".join(missing_sheet) + "\n", encoding="utf-8")

    if not complete:
        print(f"status=incomplete merged_rows={len(merged)} missing_scores={len(missing_scores)} invalid_scores={len(invalid_scores)}")
        return

    write_csv(audit_dir / "blind_internal_audit_scores_merged_private.csv", merged)
    by_method = summarize_group(merged, ["method", "method_display"])
    by_family_method = summarize_group(merged, ["family", "method", "method_display"])
    by_task_method = summarize_group(merged, ["family", "task", "method", "method_display"])
    write_csv(audit_dir / "blind_internal_audit_summary_by_method.csv", by_method)
    write_csv(audit_dir / "blind_internal_audit_summary_by_family_method.csv", by_family_method)
    write_csv(audit_dir / "blind_internal_audit_summary_by_task_method.csv", by_task_method)

    fields = [
        "method_display",
        "n",
        "edit_correct_1_5_mean",
        "relation_correct_1_5_mean",
        "source_preservation_1_5_mean",
        "locality_1_5_mean",
        "artifact_severity_1_5_mean",
        "overall_1_5_mean",
    ]
    lines = [
        "# Phase2 Blind Internal Audit Summary",
        "",
        "Internal visual audit only; not a user study.",
        "",
        "Scope: Phase2 T1-T5, three source cases per family, seeds 10/11/12.",
        "",
        markdown_table(by_method, fields),
        "",
        "Artifact severity is lower-is-better.",
    ]
    (audit_dir / "blind_internal_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("status=complete")
    print(f"wrote {audit_dir / 'blind_internal_audit_summary.md'}")


if __name__ == "__main__":
    main()
