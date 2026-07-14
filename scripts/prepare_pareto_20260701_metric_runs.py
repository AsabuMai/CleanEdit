from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJ = Path(__file__).resolve().parents[1]
EXP = "pareto_sweep_20260701"
ROOT = PROJ / "outputs" / EXP
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_pareto_non_t4_33.json"
OUT = PROJ / "outputs/pareto_metric_runs_20260701"

FLUX_TAGS = ("g070", "g085", "g100", "g115")
SCOPE_TAGS = ("g070", "g085", "g115")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        from shutil import copy2

        copy2(src, dst)


def first(pattern: str) -> Path | None:
    found = sorted(PROJ.glob(pattern))
    return found[0] if found else None


def result_path(key: str, method: str) -> Path | None:
    if method.startswith("scope_flux_"):
        tag = method.removeprefix("scope_flux_")
        return ROOT / f"scope_flux_{tag}" / key / "dece_rf_flux/seed_10/result.png"
    if method.startswith("scope_sd3_"):
        tag = method.removeprefix("scope_sd3_")
        return ROOT / f"scope_sd3_{tag}" / key / "support_v3_controller_rmsgap/seed_10/result.png"
    if method.startswith("reflex_"):
        tag = method.removeprefix("reflex_")
        return ROOT / f"reflex_{tag}" / key / "reflex/seed_10/result.png"
    if method.startswith("fireflow_"):
        tag = method.removeprefix("fireflow_")
        return ROOT / f"fireflow_{tag}" / key / "fireflow/seed_10/result.png"
    if method.startswith("flowedit_flux_"):
        tag = method.removeprefix("flowedit_flux_")
        return first(f"outputs/{EXP}/flowedit_flux_{tag}/FLUX/src_{key}/tar_*/output_*seed10.png")
    if method.startswith("flowedit_sd3_"):
        tag = method.removeprefix("flowedit_sd3_")
        return first(f"outputs/{EXP}/flowedit_sd3_{tag}/SD3/src_{key}/tar_*/output_*seed10.png")
    if method.startswith("splitflow_sd3_"):
        tag = method.removeprefix("splitflow_sd3_")
        return first(f"outputs/{EXP}/splitflow_sd3_{tag}/SD3/src_{key}/tar_*/output_*seed10.png")
    if method.startswith("otrf_sd3_"):
        tag = method.removeprefix("otrf_sd3_")
        return first(f"outputs/{EXP}/otrf_sd3_{tag}/SD3/src_{key}/tar_*/enhanced/*seed10.png")
    return None


def method_names() -> list[str]:
    methods: list[str] = []
    methods += [f"scope_flux_{tag}" for tag in SCOPE_TAGS]
    methods += [f"flowedit_flux_{tag}" for tag in FLUX_TAGS]
    methods += [f"fireflow_{tag}" for tag in FLUX_TAGS]
    methods += [f"reflex_{tag}" for tag in FLUX_TAGS]
    methods += [f"scope_sd3_{tag}" for tag in SCOPE_TAGS]
    methods += [f"flowedit_sd3_{tag}" for tag in FLUX_TAGS]
    methods += [f"splitflow_sd3_{tag}" for tag in FLUX_TAGS]
    methods += [f"otrf_sd3_{tag}" for tag in FLUX_TAGS]
    return methods


def write_run(item: dict, method: str, src: Path, out_dir: Path) -> None:
    run_dir = out_dir / item["key"] / method / "seed_10"
    run_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(src, run_dir / "result.png")
    metadata = {
        "method": method,
        "task": item["key"],
        "seed": 10,
        "image": item["image"],
        "source_image": item["image"],
        "source_prompt": item["source_prompt"],
        "target_prompt": item["target_prompt"],
        "family": item.get("family", ""),
        "family_label": item.get("family_label", ""),
        "flowedit_idx": item.get("flowedit_idx", ""),
        "original_result": str(src),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "stats.json").write_text(json.dumps({"steps": []}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(f"pareto metric wrapper linked from {src}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    methods = method_names()
    counts = {method: 0 for method in methods}
    missing: list[dict[str, str]] = []

    for item in items:
        key = item["key"]
        for method in methods:
            src = result_path(key, method)
            if src is None or not src.exists():
                missing.append({"task": key, "method": method, "path": str(src or "")})
                continue
            counts[method] += 1
            if not args.check_only:
                write_run(item, method, src, args.out_dir)

    if not args.check_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
        (args.out_dir / "_methods.txt").write_text(" ".join(methods) + "\n", encoding="utf-8")

    print("out", args.out_dir)
    print("methods", len(methods))
    print(json.dumps(counts, indent=2))
    print("missing", len(missing))
    if missing and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
