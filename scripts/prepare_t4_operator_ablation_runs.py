from __future__ import annotations

import json
import os
from pathlib import Path
from shutil import copy2


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_t4_recolor_19.json"
OUT = PROJ / "outputs/t4_operator_ablation_seed10"

SOURCES = {
    "ours_sd3_plain": PROJ / "outputs/norestore_metric_runs",
    "ours_sd3_texture": PROJ / "outputs/t4_texture_metric_runs",
    "ours_flux_plain": PROJ / "outputs/norestore_metric_runs",
    "ours_flux_texture": PROJ / "outputs/t4_texture_metric_runs",
}

SOURCE_METHOD = {
    "ours_sd3_plain": "ours_sd3",
    "ours_sd3_texture": "ours_sd3",
    "ours_flux_plain": "ours_flux",
    "ours_flux_texture": "ours_flux",
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        copy2(src, dst)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing: list[dict[str, str]] = []
    counts = {method: 0 for method in SOURCES}

    for entry in manifest:
        key = entry["key"]
        for method, root in SOURCES.items():
            src_method = SOURCE_METHOD[method]
            src = root / key / src_method / "seed_10/result.png"
            if not src.exists():
                missing.append({"task": key, "method": method, "path": str(src)})
                continue
            run = OUT / key / method / "seed_10"
            link_or_copy(src, run / "result.png")
            metadata = {
                "method": method,
                "source_method": src_method,
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
            (run / "command.txt").write_text(f"T4 operator ablation wrapper linked from {src}\n", encoding="utf-8")
            counts[method] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    print("out", OUT)
    print("counts", counts)
    print("missing", len(missing))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
