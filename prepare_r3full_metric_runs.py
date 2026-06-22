import json, os
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MAN = json.load(open(PROJ / "data/flowedit_compatible_118/manifest.json"))
OUT = PROJ / "outputs/fe118_r3full_metric_runs"
SRC = {"dece_rf_sd3": ("outputs/fe118_r3full_dece_sd3", "support_v3_controller_rmsgap"),
       "dece_rf_flux": ("outputs/fe118_r3full_dece_flux", "dece_rf_flux")}
def lc(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    try: os.symlink(src, dst)
    except OSError:
        from shutil import copy2; copy2(src, dst)
counts = {m: 0 for m in SRC}; missing = []
for item in MAN:
    key = item["key"]
    for method, (base, sub) in SRC.items():
        src = PROJ / base / key / sub / "seed_10/result.png"
        if not src.exists(): missing.append((key, method)); continue
        rd = OUT / key / method / "seed_10"; rd.mkdir(parents=True, exist_ok=True)
        lc(src, rd / "result.png")
        meta = {"method": method, "task": key, "seed": 10, "image": item["image"], "source_image": item["image"],
                "source_prompt": item["source_prompt"], "target_prompt": item["target_prompt"],
                "family": item.get("family"), "family_label": item.get("family_label")}
        (rd / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (rd / "stats.json").write_text(json.dumps({"steps": []}) + "\n", encoding="utf-8")
        (rd / "command.txt").write_text("r3full linked from " + str(src) + "\n", encoding="utf-8")
        counts[method] += 1
print("counts", counts, "missing", len(missing))
