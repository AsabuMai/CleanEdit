import csv, json
from collections import defaultdict
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
man = {e["key"]: e for e in json.load(open(PROJ / "data/flowedit_compatible_118/manifest_probe_t5t1.json"))}
def fam(t): return (man.get(t, {}).get("family_label") or "?")[:2]
csv_path = PROJ / "experiments/probe_t5t1_metrics/metrics.csv"
NUM = ["clip_direction_similarity", "local_clip_t", "local_clip_target_score", "edit_score",
       "outside_mask_l1", "bg_lpips_x100", "bg_psnr"]
HDR = ["CLIPdir", "fLocCLIP", "LocCLIP-T", "editScore", "NonEditMAE", "BGLPIPS", "BG-PSNR"]
def fnum(r, k):
    try: return float(r.get(k, "") or 0)
    except: return 0.0
rows = defaultdict(lambda: defaultdict(list))     # method -> metric -> list
percase = defaultdict(dict)                         # key -> method -> dirscore
for r in csv.DictReader(open(csv_path)):
    m = r["method"]
    for k in NUM:
        rows[m][k].append(fnum(r, k))
    percase[r["task"]][m] = fnum(r, "clip_direction_similarity")
order = ["dece_current", "dece_A", "dece_B", "sam_flow_sd3"]
print("=== PROBE T5+T1 (n=8) mean ===")
print("%-14s " % "method" + " ".join("%10s" % h for h in HDR))
for m in order:
    if m not in rows: continue
    v = rows[m]; n = len(v["edit_score"])
    print("%-14s " % m + " ".join("%10.4f" % (sum(v[k]) / len(v[k])) for k in NUM))
print("\n=== per-case CLIPdir (edit-direction; higher=better) ===")
print("%-40s %8s %8s %8s %8s" % ("case", "current", "A", "B", "samflow"))
for k in man:
    pc = percase.get(k, {})
    print("%-40s %8.3f %8.3f %8.3f %8.3f" % (k[:40],
        pc.get("dece_current", 0), pc.get("dece_A", 0), pc.get("dece_B", 0), pc.get("sam_flow_sd3", 0)))
print("\nNonEditMAE/BGLPIPS lower=better; CLIPdir/fLocCLIP/editScore/BG-PSNR higher=better.")
