from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_ablation_non_t4_116.json"
OUT = PROJ / "outputs/final_component_ablation_runs_20260630"
SEED = "10"
LINK_FILES = ("result.png", "metadata.json", "stats.json", "command.txt")

METHODS = {
    "full_sd3": (
        PROJ / "outputs/fe135_norestore_full_sd3",
        "support_v3_controller_rmsgap",
        "SPARE-RF-SD3",
    ),
    "sd3_wopreserve": (
        PROJ / "outputs/final_component_ablation_non_t4_sd3_preserve",
        "support_v3_controller_rmsgap",
        "SD3 w/o source-anchor preservation",
    ),
    "sd3_wofeedback": (
        PROJ / "outputs/final_component_ablation_non_t4_sd3_adaptive",
        "support_v3_controller_rmsgap",
        "SD3 w/o clean-gap feedback",
    ),
    "sd3_woopsupport": (
        PROJ / "outputs/final_component_ablation_non_t4_sd3_opsupport",
        "support_v3_controller_rmsgap",
        "SD3 w/o operation-specific support candidate",
    ),
    "full_flux": (
        PROJ / "outputs/fe135_norestore_full_flux",
        "dece_rf_flux",
        "SPARE-RF-FLUX",
    ),
    "flux_wopreserve": (
        PROJ / "outputs/final_component_ablation_non_t4_flux_preserve",
        "dece_rf_flux",
        "FLUX w/o source-anchor preservation",
    ),
    "flux_wofeedback": (
        PROJ / "outputs/final_component_ablation_non_t4_flux_adaptive",
        "dece_rf_flux",
        "FLUX w/o clean-gap feedback",
    ),
    "flux_woopsupport": (
        PROJ / "outputs/final_component_ablation_non_t4_flux_opsupport",
        "dece_rf_flux",
        "FLUX w/o operation-specific support candidate",
    ),
}


def reset(path: Path) -> None:
    resolved = path.resolve()
    allowed = OUT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise RuntimeError(f"refusing to clean {resolved}")
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def main() -> int:
    tasks = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reset(OUT)
    missing: list[dict[str, str]] = []
    linked = 0

    for entry in tasks:
        task = entry["key"]
        for method, (root, inner, display) in METHODS.items():
            src = root / task / inner / f"seed_{SEED}"
            dst = OUT / task / method / f"seed_{SEED}"
            if not (src / "result.png").exists():
                missing.append({"task": task, "method": method, "path": str(src)})
                continue
            for name in LINK_FILES:
                if link_or_copy(src / name, dst / name):
                    linked += 1
            metadata = {
                "method": method,
                "method_display": display,
                "task": task,
                "seed": int(SEED),
                "image": entry["image"],
                "source_image": entry["image"],
                "source_prompt": entry["source_prompt"],
                "target_prompt": entry["target_prompt"],
                "family": entry.get("family", ""),
                "family_label": entry.get("family_label", ""),
                "flowedit_idx": entry.get("flowedit_idx", ""),
                "original_run": str(src),
                "original_result": str(src / "result.png"),
            }
            (dst / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if not (dst / "stats.json").exists():
                (dst / "stats.json").write_text(json.dumps({"steps": []}, indent=2) + "\n", encoding="utf-8")
            if not (dst / "command.txt").exists():
                (dst / "command.txt").write_text(f"linked from {src}\n", encoding="utf-8")

    audit = {
        "tasks": len(tasks),
        "methods": sorted(METHODS),
        "expected_rows": len(tasks) * len(METHODS),
        "linked_files": linked,
        "missing": missing,
    }
    (OUT / "_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "missing": len(missing)}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
