from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path


REMOTE_PROJECT = Path("/cluster/users/grad/2025/25t8103/project")
PROJECT = REMOTE_PROJECT if REMOTE_PROJECT.exists() else Path(r"I:\Downloads\1\2")
EXP = PROJECT / "experiments" / "support_v3_2026-06-02" if PROJECT == REMOTE_PROJECT else PROJECT / "phase2_lock_2026-06-11"
REGISTRY = EXP / "phase2_final_selected_runs_2026-06-11.csv"
OUT_DIR = PROJECT / "outputs" / "phase2_selected_metric_refresh_20260612"
OUT_JSON = EXP / "phase2_selected_metric_refresh_manifest.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def replace_link(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link, target_is_directory=True)


def main() -> int:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    rows = [
        row
        for row in read_rows(REGISTRY)
        if row.get("metric_status") == "needs_metric_refresh"
        and row.get("method") == "support_v3_controller_rmsgap"
    ]
    if not rows:
        OUT_JSON.write_text(json.dumps({"rows": 0, "items": []}, indent=2) + "\n", encoding="utf-8")
        print(f"no refresh rows; manifest={OUT_JSON}")
        return 0

    items: list[dict[str, str]] = []
    for row in rows:
        selected = PROJECT / row["selected_result_image"]
        run_dir = selected.parent
        if not selected.exists():
            raise FileNotFoundError(selected)
        task = row["task"]
        seed = row["seed"]
        method = row["method"]
        link = OUT_DIR / task / method / f"seed_{seed}"
        replace_link(link, run_dir)
        items.append(
            {
                "task": task,
                "seed": seed,
                "method": method,
                "link_run_dir": rel(link),
                "target_run_dir": rel(run_dir),
                "selected_result_image": row["selected_result_image"],
            }
        )

    OUT_JSON.write_text(json.dumps({"rows": len(items), "items": items}, indent=2) + "\n", encoding="utf-8")
    print(f"refresh_rows={len(items)}")
    print(f"outputs_dir={OUT_DIR}")
    print(f"manifest={OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
