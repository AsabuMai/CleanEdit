import csv
import json
import os
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_118/manifest.json"
OUT = PROJ / "outputs/flowedit118_metric_runs"


METHODS = [
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
    out = {}
    if not path.exists():
        return out
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        if row.get("status") == "complete" and row.get("result_image"):
            out[row["task"]] = PROJ / row["result_image"]
    return out


def result_path(item: dict, method: str, csv_maps: dict[str, dict[str, Path]]) -> Path | None:
    key = item["key"]
    if method in csv_maps:
        return csv_maps[method].get(key)
    if method == "flowedit_flux":
        return first_png(f"_baselines/src/FlowEdit/outputs/FlowEdit_FE118_FLUX/FLUX/src_{key}/tar_*/output_*seed10.png")
    if method == "flowedit_sd3":
        return first_png(f"_baselines/src/FlowEdit/outputs/FlowEdit_FE118_SD3/SD3/src_{key}/tar_*/output_*seed10.png")
    if method == "splitflow_sd3":
        return first_png(f"_baselines/src/SplitFlow/outputs/SplitFlow_FE118_SD3/SD3/src_{key}/tar_*/output_*seed10.png")
    if method == "reflex":
        return PROJ / "outputs/flowedit_baselines/reflex" / key / "reflex/seed_10/result.png"
    if method == "sam_flow_flux":
        return PROJ / "outputs/flowedit_baselines/sam_flow_flux" / key / "sam_flow_flux/seed_10/result.png"
    if method == "sam_flow_sd3":
        return PROJ / "outputs/flowedit_baselines/sam_flow_sd3" / key / "sam_flow_sd3/seed_10/result.png"
    if method == "instruct_pix2pix":
        return PROJ / "outputs/flowedit_baselines/instruct_pix2pix" / key / "instruct_pix2pix/seed_10/result.png"
    if method == "ledits_pp":
        return PROJ / "outputs/flowedit_baselines/ledits_pp" / key / "ledits_pp/seed_10/result.png"
    return None


def write_run(item: dict, method: str, src_result: Path) -> None:
    run_dir = OUT / item["key"] / method / "seed_10"
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
        "family": item.get("family"),
        "family_label": item.get("family_label"),
        "flowedit_idx": item.get("flowedit_idx"),
        "original_result": str(src_result),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "stats.json").write_text(json.dumps({"steps": []}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(f"external baseline result linked from {src_result}\n", encoding="utf-8")


def main() -> None:
    manifest = json.load(MANIFEST.open())
    csv_maps = {
        "fireflow": csv_results(PROJ / "data/flowedit_compatible_118/baseline_fireflow.csv"),
        "rf_solver_edit": csv_results(PROJ / "data/flowedit_compatible_118/baseline_rf_solver_edit.csv"),
    }
    counts = {method: 0 for method in METHODS}
    missing = []
    for item in manifest:
        for method in METHODS:
            path = result_path(item, method, csv_maps)
            if path is None or not path.exists():
                missing.append({"task": item["key"], "method": method, "path": str(path or "")})
                continue
            write_run(item, method, path)
            counts[method] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    print("out", OUT)
    print("counts", counts)
    print("missing", len(missing), OUT / "_missing.json")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
