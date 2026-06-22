import json, os, glob
from pathlib import Path
from PIL import Image
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MAN = json.load(open(PROJ / "data/flowedit_compatible_118/manifest.json"))
OUT = PROJ / "outputs/fe118_newbaselines_metric_runs"
OTRF = PROJ / "_baselines/src/OT-RF/outputs/OTRF_SD3_FE118_ENH/SD3"
DRFS = PROJ / "_baselines/src/DeltaRectifiedFlowSampling/outputs/DRFS_SD3_FE118/SD3"
def write_meta(rd, item, method):
    meta = {"method": method, "task": item["key"], "seed": 10, "image": item["image"],
            "source_image": item["image"], "source_prompt": item["source_prompt"],
            "target_prompt": item["target_prompt"], "family": item.get("family"),
            "family_label": item.get("family_label")}
    (rd / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (rd / "stats.json").write_text(json.dumps({"steps": []}) + "\n", encoding="utf-8")
    (rd / "command.txt").write_text(method + " linked for " + item["key"] + "\n", encoding="utf-8")
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        from shutil import copy2
        copy2(src, dst)
counts = {"ot_rf": 0, "drfs": 0}
missing = []
for item in MAN:
    key = item["key"]
    # OT-RF: full result png, symlink
    g = sorted(glob.glob(str(OTRF / ("src_" + key) / "tar_0" / "enhanced" / "enhanced_*seed10.png")))
    if g:
        rd = OUT / key / "ot_rf" / "seed_10"
        rd.mkdir(parents=True, exist_ok=True)
        link(g[0], rd / "result.png")
        write_meta(rd, item, "ot_rf")
        counts["ot_rf"] += 1
    else:
        missing.append((key, "ot_rf"))
    # DRFS: 2048x1024 concat (left=source, right=edited) -> crop right half
    g2 = sorted(glob.glob(str(DRFS / ("src_" + key) / "tgt_0" / "*seed10.png")))
    if g2:
        try:
            im = Image.open(g2[0]).convert("RGB")
            w, h = im.size
            right = im.crop((w // 2, 0, w, h))
            rd = OUT / key / "drfs" / "seed_10"
            rd.mkdir(parents=True, exist_ok=True)
            right.save(rd / "result.png")
            write_meta(rd, item, "drfs")
            counts["drfs"] += 1
        except Exception as ex:
            missing.append((key, "drfs:" + repr(ex)))
    else:
        missing.append((key, "drfs"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
print("counts", counts, "missing", len(missing))
