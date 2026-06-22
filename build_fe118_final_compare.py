import csv, json
from collections import defaultdict
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
man = {e["key"]: e for e in json.load(open(PROJ / "data/flowedit_compatible_118/manifest.json"))}
def fam(t): return (man.get(t, {}).get("family_label") or "?")[:2]
SRCS = [PROJ / "experiments/flowedit118_baselines_fixedmask_metrics_20260619/metrics.csv",
        PROJ / "experiments/flowedit118_newbaselines_fixedmask_metrics_20260620/metrics.csv",
        PROJ / "experiments/flowedit118_r3full_fixedmask_metrics_20260619/metrics.csv",
        PROJ / "experiments/flowedit118_traditional_fixedmask_metrics_20260620/metrics.csv"]
MAIN = {"T1", "T2", "T4", "T5"}
# paper 8-column order
NUM = ["outside_mask_l1", "bg_psnr", "bg_lpips_x100", "bg_ssim_luma",
       "dino_source_similarity", "clip_t", "local_clip_target_score", "local_clip_t", "edit_score", "clip_direction_similarity"]
HDR = ["NonEditMAE", "BG-PSNR", "BGLPIPSx100", "BG-SSIM", "DINO-src", "CLIP-T", "LocalCLIP-T", "fLocCLIP", "editScore", "CLIPdir"]
def fnum(r, k):
    try:
        return float(r.get(k, "") or 0.0)
    except ValueError:
        return 0.0
rows = defaultdict(lambda: {"main": defaultdict(list), "t3": defaultdict(list)})
_seen=set()
for p in SRCS:
    if not p.exists():
        print("MISSING-SRC", p)
        continue
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        f = fam(r["task"])
        bucket = "main" if f in MAIN else ("t3" if f == "T3" else None)
        if bucket is None:
            continue
        _key=(r["method"], r["task"])
        if _key in _seen:
            continue
        _seen.add(_key)
        for k in NUM:
            rows[r["method"]][bucket][k].append(fnum(r, k))
order = ["instruct_pix2pix", "ledits_pp", "fireflow", "rf_solver_edit", "flowedit_flux", "flowedit_sd3", "splitflow_sd3",
         "ot_rf", "drfs", "reflex", "sam_flow_flux", "sam_flow_sd3", "dece_rf_sd3", "dece_rf_flux"]
def tbl(bucket, title):
    print("\n=== %s ===" % title)
    print("%-16s %4s " % ("method", "n") + " ".join("%11s" % h for h in HDR))
    for m in order:
        d = rows.get(m, {}).get(bucket)
        if not d or not d["edit_score"]:
            continue
        n = len(d["edit_score"])
        v = lambda k: sum(d[k]) / len(d[k])
        print("%-16s %4d " % (m, n) + " ".join("%11.4f" % v(k) for k in NUM))
tbl("main", "MAIN  T1+T2+T4+T5  (excl. T3 text) -- paper 8 metrics")
tbl("t3", "T3  surface_decal (TEXT - reported separately)")
print("\nNote: NonEditMAE/BGLPIPSx100 lower=better; others higher=better.")
print("editScore and CLIP-T are full-image; LocalCLIP-T is the localized edit metric the paper leans on.")
print("ot_rf=OT-RF(WACV26), drfs=DRFS(CVPR26) are newly added SD3 RF baselines.")
