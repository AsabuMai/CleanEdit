from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
DEFAULT_MANIFEST = PROJ / "data/flowedit_compatible_135/manifest.json"
DEFAULT_OUT = PROJ / "outputs/flowedit135_metric_runs"

METHODS = [
    "ours_sd3",
    "fireflow",
    "rf_solver_edit",
    "flowedit_flux",
    "flowedit_sd3",
    "splitflow_sd3",
    "reflex",
    "sam_flow_flux",
    "sam_flow_sd3",
    "instruct_pix2pix",
    "ledits_pp",
    "otrf_enh_sd3",
    "drfs_sd3",
]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        from shutil import copy2

        copy2(src, dst)


def first_png(pattern: str) -> Path | None:
    found = sorted(PROJ.glob(pattern))
    return found[0] if found else None


def csv_results(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result = row.get("result_image", "")
            if row.get("status") == "complete" and result:
                out[row["task"]] = Path(result) if result.startswith("/") else PROJ / result
    return out


def direct_run_result(root: str, key: str, method: str) -> Path:
    return PROJ / root / key / method / "seed_10/result.png"


def result_path(item: dict, method: str, csv_maps: dict[str, dict[str, Path]]) -> Path | None:
    key = item["key"]
    if method in csv_maps:
        return csv_maps[method].get(key)
    if method == "ours_sd3":
        return PROJ / "outputs/fe135_full_dece_sd3" / key / "support_v3_controller_rmsgap/seed_10/result.png"
    if method == "reflex":
        return direct_run_result("outputs/flowedit135_baselines/reflex", key, "reflex")
    if method == "sam_flow_flux":
        return direct_run_result("outputs/flowedit135_baselines/sam_flow_flux", key, "sam_flow_flux")
    if method == "sam_flow_sd3":
        return direct_run_result("outputs/flowedit135_baselines/sam_flow_sd3", key, "sam_flow_sd3")
    if method == "instruct_pix2pix":
        return direct_run_result("outputs/flowedit135_traditional_baselines/instruct_pix2pix", key, "instruct_pix2pix")
    if method == "ledits_pp":
        return direct_run_result("outputs/flowedit135_traditional_baselines/ledits_pp", key, "ledits_pp")
    if method == "flowedit_flux":
        return first_png(f"_baselines/src/FlowEdit/outputs/FlowEdit_FE135_FLUX/FLUX/src_{key}/tar_*/output_*seed10.png")
    if method == "flowedit_sd3":
        return first_png(f"_baselines/src/FlowEdit/outputs/FlowEdit_FE135_SD3/SD3/src_{key}/tar_*/output_*seed10.png")
    if method == "splitflow_sd3":
        return first_png(f"_baselines/src/SplitFlow/outputs/SplitFlow_FE135_SD3/SD3/src_{key}/tar_*/output_*seed10.png")
    if method == "otrf_enh_sd3":
        return first_png(f"_baselines/src/OT-RF/outputs/OTRF_SD3_FE135_ENH/SD3/src_{key}/tar_*/enhanced/*seed10.png")
    if method == "drfs_sd3":
        return first_png(f"_baselines/src/DeltaRectifiedFlowSampling/outputs/DRFS_SD3_FE135/SD3/src_{key}/tgt_*/*seed10.png")
    return None


def write_run(item: dict, method: str, src_result: Path, out_dir: Path) -> None:
    run_dir = out_dir / item["key"] / method / "seed_10"
    run_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(src_result, run_dir / "result.png")
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
        "original_result": str(src_result),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "stats.json").write_text(json.dumps({"steps": []}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(f"FE135 metric wrapper linked from {src_result}\n", encoding="utf-8")


def parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return METHODS
    selected = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    unknown = [item for item in selected if item not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}; valid: {METHODS}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified FE135 metric runs from completed method outputs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only verify source outputs; do not write metric runs.")
    args = parser.parse_args()

    manifest = json.load(args.manifest.open(encoding="utf-8"))
    methods = parse_methods(args.methods)
    csv_maps = {
        "fireflow": csv_results(PROJ / "data/flowedit_compatible_135/baseline_fireflow.csv"),
        "rf_solver_edit": csv_results(PROJ / "data/flowedit_compatible_135/baseline_rf_solver_edit.csv"),
    }
    counts = {method: 0 for method in methods}
    missing = []
    for item in manifest:
        for method in methods:
            path = result_path(item, method, csv_maps)
            if path is None or not path.exists():
                missing.append({"task": item["key"], "method": method, "path": str(path or "")})
                continue
            if not args.check_only:
                write_run(item, method, path, args.out_dir)
            counts[method] += 1

    if not args.check_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    print("out", args.out_dir)
    print("methods", methods)
    print("counts", counts)
    print("missing", len(missing), args.out_dir / "_missing.json")
    if missing and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
