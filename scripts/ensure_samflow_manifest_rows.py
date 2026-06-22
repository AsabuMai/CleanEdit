#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import runpy
from pathlib import Path


FIELDS = [
    "baseline",
    "task",
    "seed",
    "status",
    "source_image",
    "source_prompt",
    "target_prompt",
    "result_image",
    "metadata",
    "command",
    "matched_conditions",
    "failure_reason",
    "notes",
]

SAM_BASELINES = ("sam_flow_sd3", "sam_flow_flux")


def load_module(path: Path) -> dict:
    return runpy.run_path(str(path))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def notes_for(base: dict, baseline: str) -> str:
    item = base["BASELINES"][baseline]
    if isinstance(item, dict):
        return "; ".join([item["paper_bucket"], item["backbone"], item["runner"]])
    bucket, backbone = item
    return f"{bucket}; {backbone}; scripts/run_samflow_baseline.py --baselines {baseline}"


def ensure_rows(
    *,
    manifest: Path,
    module_path: Path,
    tasks: list[str],
    seeds: list[str],
    baselines: list[str],
) -> tuple[int, int]:
    base = load_module(module_path)
    task_specs = base["TASKS"]
    existing = read_rows(manifest)
    seen = {
        (row.get("baseline", ""), row.get("task", ""), str(row.get("seed", "")).removeprefix("seed_"))
        for row in existing
    }
    added = 0
    for baseline in baselines:
        if baseline not in base["BASELINES"]:
            raise SystemExit(f"baseline {baseline} is not registered in {module_path}")
        for task in tasks:
            if task not in task_specs:
                raise SystemExit(f"task {task} is not registered in {module_path}")
            spec = task_specs[task]
            for seed in seeds:
                key = (baseline, task, seed)
                if key in seen:
                    continue
                existing.append(
                    {
                        "baseline": baseline,
                        "task": task,
                        "seed": seed,
                        "status": "pending",
                        "source_image": spec["source_image"],
                        "source_prompt": spec["source_prompt"],
                        "target_prompt": spec["target_prompt"],
                        "result_image": "",
                        "metadata": "",
                        "command": "",
                        "matched_conditions": "",
                        "failure_reason": "",
                        "notes": notes_for(base, baseline),
                    }
                )
                seen.add(key)
                added += 1
    write_rows(manifest, existing)
    return added, len(existing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--scope", choices=["t1_t4", "t5", "all"], default="all")
    parser.add_argument("--baselines", default=" ".join(SAM_BASELINES))
    parser.add_argument("--seeds", default="10 11 12")
    args = parser.parse_args()

    project = args.project
    scripts = project / "scripts"
    exp = project / "experiments" / "support_v3_2026-06-02"
    baselines = [item for item in args.baselines.split() if item]
    seeds = [item.removeprefix("seed_") for item in args.seeds.split() if item]

    total_added = 0
    if args.scope in {"t1_t4", "all"}:
        module_path = scripts / "init_e2_t1_t4_formal_manifest.py"
        base = load_module(module_path)
        tasks = list(base["TASKS"].keys())
        added, rows = ensure_rows(
            manifest=exp / "e2_t1_t4_formal_baseline_manifest.csv",
            module_path=module_path,
            tasks=tasks,
            seeds=seeds,
            baselines=baselines,
        )
        total_added += added
        print(f"t1_t4 added={added} rows={rows}")
    if args.scope in {"t5", "all"}:
        module_path = scripts / "init_e2_t5_formal_manifest.py"
        base = load_module(module_path)
        tasks = list(base["FORMAL_T5_TASKS"])
        added, rows = ensure_rows(
            manifest=exp / "e2_t5_formal_baseline_manifest.csv",
            module_path=module_path,
            tasks=tasks,
            seeds=seeds,
            baselines=baselines,
        )
        total_added += added
        print(f"t5 added={added} rows={rows}")
    print(f"total_added={total_added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
