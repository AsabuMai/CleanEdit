from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest.json"
T4_MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_t4_recolor_19.json"
OUT = PROJ / "outputs/final_multiseed_metric_runs_20260630"
SEEDS = ("10", "11", "12")
LINK_FILES = ("result.png", "metadata.json", "stats.json", "command.txt")


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


def source_dir(task: str, method: str, seed: str, t4_tasks: set[str]) -> tuple[Path, str]:
    if task in t4_tasks:
        if method == "ours_sd3":
            return (
                PROJ / "outputs/fe135_t4_texture_sd3" / task / "support_v3_controller_rmsgap" / f"seed_{seed}",
                "t4_texture_sd3",
            )
        return (
            PROJ / "outputs/fe135_t4_texture_flux_h100" / task / "dece_rf_flux" / f"seed_{seed}",
            "t4_texture_flux",
        )
    if method == "ours_sd3":
        return (
            PROJ / "outputs/fe135_norestore_full_sd3" / task / "support_v3_controller_rmsgap" / f"seed_{seed}",
            "norestore_sd3",
        )
    return (
        PROJ / "outputs/fe135_norestore_full_flux" / task / "dece_rf_flux" / f"seed_{seed}",
        "norestore_flux",
    )


def main() -> int:
    tasks = json.loads(MANIFEST.read_text(encoding="utf-8"))
    t4_tasks = {entry["key"] for entry in json.loads(T4_MANIFEST.read_text(encoding="utf-8"))}
    reset(OUT)
    missing: list[dict[str, str]] = []
    linked = 0
    for entry in tasks:
        task = entry["key"]
        for method in ("ours_sd3", "ours_flux"):
            for seed in SEEDS:
                src, kind = source_dir(task, method, seed, t4_tasks)
                dst = OUT / task / method / f"seed_{seed}"
                if not (src / "result.png").exists():
                    missing.append({"task": task, "method": method, "seed": seed, "kind": kind, "path": str(src)})
                    continue
                for name in LINK_FILES:
                    if link_or_copy(src / name, dst / name):
                        linked += 1
                if not (dst / "metadata.json").exists():
                    metadata = {
                        "method": method,
                        "task": task,
                        "seed": int(seed),
                        "image": entry["image"],
                        "source_image": entry["image"],
                        "source_prompt": entry["source_prompt"],
                        "target_prompt": entry["target_prompt"],
                        "family": entry.get("family", ""),
                        "family_label": entry.get("family_label", ""),
                        "original_result": str((src / "result.png").resolve()),
                    }
                    (dst / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                if not (dst / "stats.json").exists():
                    (dst / "stats.json").write_text(json.dumps({"steps": []}, indent=2) + "\n", encoding="utf-8")
                if not (dst / "command.txt").exists():
                    (dst / "command.txt").write_text(f"linked from {src}\n", encoding="utf-8")
    (OUT / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    (OUT / "_README.txt").write_text(
        f"tasks={len(tasks)}\nmethods=2\nseeds={','.join(SEEDS)}\nlinked={linked}\nmissing={len(missing)}\n",
        encoding="utf-8",
    )
    print("out", OUT)
    print("linked", linked)
    print("missing", len(missing))
    if missing:
        for row in missing[:20]:
            print("MISSING", row)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
