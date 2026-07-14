from __future__ import annotations

import json
import os
from pathlib import Path
from shutil import copy2


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_t4_recolor_19.json"
OUT = PROJ / "outputs/t4_texture_metric_runs"

METHOD_SOURCES = {
    "ours_sd3": (
        PROJ / "outputs/fe135_t4_texture_sd3",
        "support_v3_controller_rmsgap/seed_10/result.png",
    ),
    "ours_flux": (
        PROJ / "outputs/fe135_t4_texture_flux_h100",
        "dece_rf_flux/seed_10/result.png",
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


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts = {method: 0 for method in METHOD_SOURCES}
    missing: list[dict[str, str]] = []
    for entry in manifest:
        key = entry["key"]
        for method, (root, suffix) in METHOD_SOURCES.items():
            src = root / key / suffix
            if not src.exists():
                missing.append({"task": key, "method": method, "path": str(src)})
                continue
            run = OUT / key / method / "seed_10"
            run.mkdir(parents=True, exist_ok=True)
            link_or_copy(src, run / "result.png")
            metadata = {
                "method": method,
                "task": key,
                "seed": 10,
                "image": entry["image"],
                "source_image": entry["image"],
                "source_prompt": entry["source_prompt"],
                "target_prompt": entry["target_prompt"],
                "family": entry.get("family", ""),
                "family_label": entry.get("family_label", ""),
                "flowedit_idx": entry.get("flowedit_idx", ""),
                "original_result": str(src),
            }
            (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            (run / "stats.json").write_text(json.dumps({"steps": []}, indent=2) + "\n", encoding="utf-8")
            (run / "command.txt").write_text(f"T4 texture metric wrapper linked from {src}\n", encoding="utf-8")
            counts[method] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    print("out", OUT)
    print("counts", counts)
    print("missing", len(missing))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
