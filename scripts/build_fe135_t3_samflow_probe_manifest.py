from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REMOTE_PREFIX = "/cluster/users/grad/2025/25t8103/project/"
FIELDS = [
    "baseline",
    "task",
    "seed",
    "status",
    "source_image",
    "source_prompt",
    "target_prompt",
    "source_tokens",
    "target_tokens",
    "unchanged_tokens",
    "result_image",
    "metadata",
    "command",
    "matched_conditions",
    "failure_reason",
    "notes",
]
METHOD_TO_SAMFLOW = {
    "sd3": "sam_flow_sd3",
    "flux": "sam_flow_flux",
}


def rel_project(path: str) -> str:
    if path.startswith(REMOTE_PREFIX):
        return path[len(REMOTE_PREFIX) :]
    return path


def read_audit(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build FE135 failed-only T3 Sam-Flow probe manifest.")
    ap.add_argument("--audit", type=Path, default=Path("fe135_strict_review_20260626_seed10/strict_visual_audit_final.csv"))
    ap.add_argument("--manifest", type=Path, default=Path("remote_patch/manifest_sam_135.json"))
    ap.add_argument("--out", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v1_manifest.csv"))
    ap.add_argument("--methods", default="sd3 flux")
    ap.add_argument("--keys", default="", help="Optional whitespace/comma-separated key filter.")
    ap.add_argument("--seed", default="10")
    args = ap.parse_args()

    methods = {item.strip() for item in args.methods.replace(",", " ").split() if item.strip()}
    keys = {item.strip() for item in args.keys.replace(",", " ").split() if item.strip()}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_key = {item["key"]: item for item in manifest}
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for audit_row in read_audit(args.audit):
        method = audit_row.get("method", "")
        key = audit_row.get("key", "")
        if method not in methods:
            continue
        if keys and key not in keys:
            continue
        if audit_row.get("family") != "T3_surface_decal":
            continue
        if audit_row.get("status") != "fail":
            continue
        item = by_key.get(key)
        if not item:
            raise SystemExit(f"missing key in manifest: {key}")
        baseline = METHOD_TO_SAMFLOW[method]
        ident = (baseline, key, args.seed)
        if ident in seen:
            continue
        seen.add(ident)
        rows.append(
            {
                "baseline": baseline,
                "task": key,
                "seed": args.seed,
                "status": "pending",
                "source_image": rel_project(item["image"]),
                "source_prompt": item.get("source_prompt", ""),
                "target_prompt": item.get("target_prompt", ""),
                "source_tokens": str(item.get("host_tokens", "")).lower(),
                "target_tokens": str(item.get("new_tokens", "")).lower(),
                "unchanged_tokens": "",
                "result_image": "",
                "metadata": "",
                "command": "",
                "matched_conditions": "",
                "failure_reason": "",
                "notes": f"FE135 T3 failed-only Sam-Flow probe for original {method}; separate probe, not final candidate.",
            }
        )

    rows.sort(key=lambda row: (row["baseline"], row["task"]))
    write_rows(args.out, rows)
    print(args.out)
    print("rows", len(rows))
    print("sam_flow_sd3", sum(1 for row in rows if row["baseline"] == "sam_flow_sd3"))
    print("sam_flow_flux", sum(1 for row in rows if row["baseline"] == "sam_flow_flux"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
