#!/usr/bin/env python3
"""Verify provenance, locked inputs, and no-final eligibility for all nine buckets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    lock_sha = file_sha(args.lock)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for bucket, bucket_spec in lock["buckets"].items():
        manifest = args.lock.parents[3] / bucket_spec["manifest"]
        if file_sha(manifest) != bucket_spec["manifest_sha256"]:
            errors.append(f"{bucket}: manifest hash mismatch")
        for asset in bucket_spec.get("recipe_assets", {}).values():
            path = Path(asset["path"])
            if not path.is_file() or path.stat().st_size != asset["size_bytes"] or file_sha(path) != asset["sha256"]:
                errors.append(f"{bucket}: recipe asset mismatch: {path}")
        for key, case in bucket_spec["cases"].items():
            for asset in case.get("support_assets", {}).values():
                path = Path(asset["path"])
                if not path.is_file() or path.stat().st_size != asset["size_bytes"] or file_sha(path) != asset["sha256"]:
                    errors.append(f"{bucket}/{key}: support asset mismatch: {path}")
            run_dir = args.output_root / bucket / key / "dece_rf_flux" / "seed_10"
            metadata_path = run_dir / "metadata.json"
            result_path = run_dir / "result.png"
            if not metadata_path.is_file() or not result_path.is_file():
                errors.append(f"{bucket}/{key}: missing result or metadata")
                continue
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            checks = {
                "seed": data.get("seed") == 10,
                "steps": data.get("num_inference_steps") == 12,
                "n_max": data.get("n_max") == 10,
                "purpose": data.get("run_purpose") == "evaluation",
                "protocol": data.get("comparison_protocol") == "single_pass",
                "postprocess_mode": data.get("final_postprocess_mode") == "none",
                "mask_blend": float(data.get("final_mask_blend_scale", -1)) == 0.0,
                "recolor_blend": float(data.get("final_recolor_blend_scale", -1)) == 0.0,
                "postprocess_applied": data.get("postprocess_applied") is False,
                "eligible": data.get("evaluation_eligible") is True,
                "eligible_reasons": data.get("eligibility_reasons") == [],
                "git_clean": data.get("git_dirty") is False,
                "commit": data.get("git_commit") == args.expected_commit,
                "bucket": data.get("evidence_bucket") == bucket,
                "recipe": data.get("evidence_recipe") == bucket_spec["recipe"],
                "lock": data.get("evidence_lock_sha256") == lock_sha,
                "config_hash": config_sha(data.get("run_config", {})) == data.get("run_config_sha256"),
                "source_path": data.get("source_image_path") == case["source_image_path"],
                "source_size": data.get("source_image_size_bytes") == case["source_image_size_bytes"],
                "source_hash": data.get("source_image_sha256") == case["source_image_sha256"],
            }
            failed = [name for name, ok in checks.items() if not ok]
            if failed:
                errors.append(f"{bucket}/{key}: failed={failed}")
            rows.append({
                "bucket": bucket,
                "key": key,
                "recipe": bucket_spec["recipe"],
                "metadata": str(metadata_path),
                "result": str(result_path),
                "config_sha256": data.get("run_config_sha256"),
                "source_image_sha256": data.get("source_image_sha256"),
                "eligible": data.get("evaluation_eligible"),
                "failed_checks": failed,
            })

    expected = sum(item["expected_count"] for item in lock["buckets"].values())
    if len(rows) != expected:
        errors.append(f"verified row count {len(rows)} != expected {expected}")
    report = {
        "lock_sha256": lock_sha,
        "expected_commit": args.expected_commit,
        "expected_runs": expected,
        "verified_runs": len(rows),
        "passed": not errors,
        "errors": errors,
        "runs": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"verified {len(rows)} clean, eligible, locked no-final FLUX runs")


if __name__ == "__main__":
    main()
