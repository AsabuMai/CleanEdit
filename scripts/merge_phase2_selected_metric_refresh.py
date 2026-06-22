from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


REMOTE_PROJECT = Path("/cluster/users/grad/2025/25t8103/project")
PROJECT = REMOTE_PROJECT if REMOTE_PROJECT.exists() else Path(r"I:\Downloads\1\2")
EXP = PROJECT / "experiments" / "support_v3_2026-06-02" if PROJECT == REMOTE_PROJECT else PROJECT / "phase2_lock_2026-06-11"
BASE = EXP / "phase2_internal_bg_metrics.csv"
REFRESH = EXP / "phase2_selected_metric_refresh_metrics.csv"
OUT_AUDIT = EXP / "phase2_selected_metric_refresh_merge_audit.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("task", ""), row.get("seed", "").removeprefix("seed_"), row.get("method", ""))


def main() -> int:
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    if not REFRESH.exists():
        raise FileNotFoundError(REFRESH)

    base_rows = read_rows(BASE)
    refresh_rows = [row for row in read_rows(REFRESH) if row.get("complete") == "True"]
    refresh_by_key = {key(row): row for row in refresh_rows}
    if len(refresh_by_key) != len(refresh_rows):
        raise RuntimeError("duplicate keys in refresh metrics")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BASE.with_name(f"{BASE.stem}.before_selected_refresh_{stamp}.csv")
    backup.write_text(BASE.read_text(encoding="utf-8"), encoding="utf-8")

    replaced: list[dict[str, str]] = []
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in base_rows:
        row_key = key(row)
        if row_key in refresh_by_key:
            merged.append(refresh_by_key[row_key])
            replaced.append({"task": row_key[0], "seed": row_key[1], "method": row_key[2]})
            seen.add(row_key)
        else:
            merged.append(row)

    missing_in_base = sorted(set(refresh_by_key) - seen)
    if missing_in_base:
        raise RuntimeError(f"refresh keys missing in base metrics: {missing_in_base}")

    fieldnames: list[str] = []
    for row in [*base_rows, *refresh_rows]:
        for name in row.keys():
            if name not in fieldnames:
                fieldnames.append(name)
    normalized = [{name: row.get(name, "") for name in fieldnames} for row in merged]
    write_rows(BASE, normalized, fieldnames)

    audit = {
        "base": str(BASE),
        "refresh": str(REFRESH),
        "backup": str(backup),
        "base_rows": len(base_rows),
        "refresh_rows": len(refresh_rows),
        "replaced_rows": len(replaced),
        "replaced": replaced,
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"replaced_rows={len(replaced)}")
    print(f"backup={backup}")
    print(f"audit={OUT_AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
