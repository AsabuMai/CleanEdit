from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from shutil import copy2


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
DEFAULT_MANIFEST = PROJ / "data/flowedit_compatible_135/manifest.json"
DEFAULT_OUT = PROJ / "outputs/flowedit135_multiseed_metric_runs"
METHOD_SOURCES = {
    "ours_sd3": (
        PROJ / "outputs/fe135_subjectpreserve_full_v10_sd3",
        "support_v3_controller_rmsgap",
    ),
    "ours_flux": (
        PROJ / "outputs/fe135_full_dece_flux_v11c_pcie8_h100",
        "dece_rf_flux",
    ),
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        copy2(src, dst)


def parse_list(value: str, valid: set[str] | None = None) -> list[str]:
    items = [x.strip() for x in value.replace(",", " ").split() if x.strip()]
    if valid is not None:
        unknown = [x for x in items if x not in valid]
        if unknown:
            raise SystemExit(f"unknown items: {unknown}; valid: {sorted(valid)}")
    return items


def wrapper_metadata(item: dict, method: str, seed: int, src_dir: Path) -> dict:
    metadata = {
        "method": method,
        "task": item["key"],
        "seed": seed,
        "image": item["image"],
        "source_image": item["image"],
        "source_prompt": item["source_prompt"],
        "target_prompt": item["target_prompt"],
        "family": item.get("family", ""),
        "family_label": item.get("family_label", ""),
        "flowedit_idx": item.get("flowedit_idx", ""),
        "original_run": str(src_dir),
        "original_result": str(src_dir / "result.png"),
    }
    original_meta = src_dir / "metadata.json"
    if original_meta.exists():
        try:
            raw = json.loads(original_meta.read_text(encoding="utf-8"))
            for key in ("runtime_seconds", "peak_gpu_memory_gb"):
                if key in raw:
                    metadata[key] = raw[key]
        except Exception:
            pass
    return metadata


def write_run(item: dict, method: str, seed: int, out_dir: Path) -> bool:
    root, inner_method = METHOD_SOURCES[method]
    src_dir = root / item["key"] / inner_method / f"seed_{seed}"
    src_result = src_dir / "result.png"
    if not src_result.exists():
        return False

    run_dir = out_dir / item["key"] / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(src_result, run_dir / "result.png")

    metadata = wrapper_metadata(item, method, seed, src_dir)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    src_stats = src_dir / "stats.json"
    if src_stats.exists():
        link_or_copy(src_stats, run_dir / "stats.json")
    else:
        (run_dir / "stats.json").write_text(
            json.dumps({"steps": []}, indent=2) + "\n",
            encoding="utf-8",
        )
    (run_dir / "command.txt").write_text(
        f"FE135 multi-seed metric wrapper linked from {src_dir}\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Ours FE135 multi-seed metric runs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", default="ours_sd3,ours_flux")
    parser.add_argument("--seeds", default="10,11,12")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    methods = parse_list(args.methods, set(METHOD_SOURCES))
    seeds = [int(x.removeprefix("seed_")) for x in parse_list(args.seeds)]

    counts = {(method, seed): 0 for method in methods for seed in seeds}
    missing = []
    for item in manifest:
        for method in methods:
            root, inner_method = METHOD_SOURCES[method]
            for seed in seeds:
                src_dir = root / item["key"] / inner_method / f"seed_{seed}"
                ok = (src_dir / "result.png").exists()
                if ok:
                    counts[(method, seed)] += 1
                    if not args.check_only:
                        write_run(item, method, seed, args.out_dir)
                else:
                    missing.append(
                        {
                            "task": item["key"],
                            "method": method,
                            "seed": seed,
                            "path": str(src_dir / "result.png"),
                        }
                    )

    if not args.check_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "_missing.json").write_text(
            json.dumps(missing, indent=2) + "\n",
            encoding="utf-8",
        )

    print("out", args.out_dir)
    print("methods", methods)
    print("seeds", seeds)
    print("counts", {f"{m}/seed_{s}": n for (m, s), n in sorted(counts.items())})
    print("missing", len(missing), args.out_dir / "_missing.json")
    if missing and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
