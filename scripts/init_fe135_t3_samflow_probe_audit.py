from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "method",
    "key",
    "family",
    "target_prompt",
    "result_path",
    "status",
    "failure_type",
    "note",
    "repair_action",
]


REMOTE_PREFIX = "/cluster/users/grad/2025/25t8103/project/"


def rel_project(path: str) -> Path:
    if path.startswith(REMOTE_PREFIX):
        return Path(path[len(REMOTE_PREFIX) :])
    return Path(path)


def map_project_path(project_tree: Path, path: str) -> Path:
    if not path:
        return Path()
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    return project_tree / rel_project(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Initialize strict visual audit CSV for FE135 T3 Sam-Flow probe outputs.")
    ap.add_argument("--project-tree", type=Path, required=True)
    ap.add_argument("--probe-manifest", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v1_manifest.csv"))
    ap.add_argument("--fe-manifest", type=Path, default=Path("remote_patch/manifest_sam_135.json"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/fe135_t3_samflow_probe_v1"))
    ap.add_argument("--method-suffix", default="probe_v1")
    ap.add_argument("--out", type=Path, default=Path("fe135_t3_samflow_probe_v1_review/strict_visual_audit_probe.csv"))
    args = ap.parse_args()

    probe_rows = list(csv.DictReader(args.probe_manifest.open(newline="", encoding="utf-8")))
    manifest = json.loads(args.fe_manifest.read_text(encoding="utf-8"))
    by_key = {item["key"]: item for item in manifest}
    out_rows: list[dict[str, str]] = []

    for row in probe_rows:
        key = row["task"]
        item = by_key[key]
        result_path = map_project_path(args.project_tree, row.get("result_image", ""))
        if not result_path:
            result_path = args.project_tree / args.output_root / row["baseline"] / key / f"seed_{row['seed']}" / "result.png"
        exists = result_path.exists()
        out_rows.append(
            {
                "method": f"{row['baseline']}_{args.method_suffix}",
                "key": key,
                "family": "T3_surface_decal",
                "target_prompt": item.get("target_prompt", row.get("target_prompt", "")),
                "result_path": str(result_path),
                "status": "fail",
                "failure_type": "unreviewed" if exists else "missing_output",
                "note": "initialized fail by default; set pass only after strict visual inspection" if exists else "probe output missing",
                "repair_action": "strict_visual_review_required" if exists else "run_or_fetch_probe_output",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.out)
    print("rows", len(out_rows))
    print("missing_output", sum(1 for row in out_rows if row["failure_type"] == "missing_output"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
