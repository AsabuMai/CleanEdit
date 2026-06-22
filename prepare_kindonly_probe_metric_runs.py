import json
import os
from pathlib import Path

PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MAN = json.load(open(PROJ / "data/flowedit_compatible_118/manifest_repair_probe.json"))
OUT = PROJ / "outputs/fe118_kindonly_probe_metric_runs"
SRC = {
    "dece_rf_flux_kindonly": ("outputs/fe118_kindonly_probe_dece_flux", "dece_rf_flux"),
    "dece_rf_sd3_kindonly": ("outputs/fe118_kindonly_probe_dece_sd3", "support_v3_controller_rmsgap"),
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        from shutil import copy2

        copy2(src, dst)


counts = {method: 0 for method in SRC}
missing = []
for item in MAN:
    key = item["key"]
    for method, (base, sub) in SRC.items():
        src = PROJ / base / key / sub / "seed_10/result.png"
        if not src.exists():
            missing.append({"task": key, "method": method, "path": str(src)})
            continue
        run_dir = OUT / key / method / "seed_10"
        link_or_copy(src, run_dir / "result.png")
        meta = {
            "method": method,
            "task": key,
            "seed": 10,
            "image": item["image"],
            "source_image": item["image"],
            "source_prompt": item["source_prompt"],
            "target_prompt": item["target_prompt"],
            "family": item.get("family"),
            "family_label": item.get("family_label"),
            "kindonly_probe": True,
        }
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (run_dir / "stats.json").write_text(json.dumps({"steps": []}, indent=2) + "\n", encoding="utf-8")
        (run_dir / "command.txt").write_text(f"kind-only probe linked from {src}\n", encoding="utf-8")
        counts[method] += 1
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
print("counts", counts, "missing", len(missing), OUT)
if missing:
    raise SystemExit(1)
