from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


VALID_STATUS = {"pass", "fail"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize strict visual audit CSV for FE135 T3 Sam-Flow probe outputs.")
    ap.add_argument("--audit", type=Path, default=Path("fe135_t3_samflow_probe_v1_review/strict_visual_audit_probe.csv"))
    ap.add_argument("--out", type=Path, default=Path("fe135_t3_samflow_probe_v1_review/strict_visual_audit_probe_summary.md"))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.audit.open(newline="", encoding="utf-8")))
    invalid = sorted({row.get("status", "") for row in rows if row.get("status", "") not in VALID_STATUS})
    if invalid:
        raise SystemExit(f"invalid status values: {invalid}")

    missing = [row for row in rows if not Path(row["result_path"]).exists()]
    status_counts = Counter(row["status"] for row in rows)
    method_status = Counter((row["method"], row["status"]) for row in rows)
    failures = Counter(row["failure_type"] or "unspecified" for row in rows if row["status"] == "fail")

    lines = [
        "# FE135 T3 Sam-Flow Probe Strict Audit Summary",
        "",
        f"- Rows: {len(rows)}",
        f"- Pass: {status_counts.get('pass', 0)}",
        f"- Fail: {status_counts.get('fail', 0)}",
        f"- Missing result paths: {len(missing)}",
        "",
        "## By Method",
        "",
    ]
    for (method, status), count in sorted(method_status.items()):
        lines.append(f"- {method} / {status}: {count}")
    lines.extend(["", "## Failure Types", ""])
    for failure_type, count in sorted(failures.items()):
        lines.append(f"- {failure_type}: {count}")
    if missing:
        lines.extend(["", "## Missing Outputs", ""])
        for row in missing[:80]:
            lines.append(f"- {row['method']} {row['key']}: `{row['result_path']}`")
        if len(missing) > 80:
            lines.append(f"- ... {len(missing) - 80} more")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)
    print("rows", len(rows))
    print("pass", status_counts.get("pass", 0))
    print("fail", status_counts.get("fail", 0))
    print("missing", len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
