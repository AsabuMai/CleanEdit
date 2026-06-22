import json, os
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_118/manifest_probe.json"
OUT = PROJ / "outputs/fe118_boost_metric_runs"
SRC = {"dece_rf_sd3": ("outputs/fe118_boost_dece_sd3", "support_v3_controller_rmsgap"),
       "dece_rf_flux": ("outputs/fe118_boost_dece_flux", "dece_rf_flux")}
def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        from shutil import copy2
        copy2(src, dst)
def main():
    man = json.load(MANIFEST.open())
    counts = {m: 0 for m in SRC}
    missing = []
    for item in man:
        key = item["key"]
        for method, (base, sub) in SRC.items():
            src = PROJ / base / key / sub / "seed_10/result.png"
            if not src.exists():
                missing.append({"task": key, "method": method}); continue
            run_dir = OUT / key / method / "seed_10"
            run_dir.mkdir(parents=True, exist_ok=True)
            link_or_copy(src, run_dir / "result.png")
            meta = {"method": method, "task": key, "seed": 10, "image": item["image"],
                    "source_image": item["image"], "source_prompt": item["source_prompt"],
                    "target_prompt": item["target_prompt"], "family": item.get("family"),
                    "family_label": item.get("family_label")}
            (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            (run_dir / "stats.json").write_text(json.dumps({"steps": []}) + "\n", encoding="utf-8")
            (run_dir / "command.txt").write_text("boost probe linked from " + str(src) + "\n", encoding="utf-8")
            counts[method] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    print("counts", counts, "missing", len(missing))
if __name__ == "__main__":
    main()
