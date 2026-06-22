import csv
from collections import defaultdict
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
SRCS = [PROJ / "experiments/flowedit118_baselines_fixedmask_metrics_20260619/metrics.csv",
        PROJ / "experiments/flowedit118_dece_fixedmask_metrics_20260619/metrics.csv"]
NUM = ["clip_direction_similarity", "edit_score", "inside_l1", "bg_l1",
       "bg_ssim_luma", "bg_lpips", "bg_dino_source", "lpips_full",
       "source_l1", "source_ssim_luma"]
def fnum(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0
groups = defaultdict(list)
for src in SRCS:
    if not src.exists():
        print("MISSING", src); continue
    for r in csv.DictReader(src.open(newline="", encoding="utf-8")):
        groups[r["method"]].append(r)
summary = []
for method, items in groups.items():
    rec = {"method": method, "n": len(items)}
    for k in NUM:
        rec[k] = sum(fnum(r, k) for r in items) / max(1, len(items))
    summary.append(rec)
# order: baselines alpha, then dece last
summary.sort(key=lambda r: (r["method"].startswith("dece"), r["method"]))
out = PROJ / "experiments/flowedit118_dece_fixedmask_metrics_20260619/comparison_v2_masked.csv"
with out.open("w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=["method", "n"] + NUM); w.writeheader(); w.writerows(summary)
print("WROTE", out)
hdr = "%-16s %4s %8s %9s %8s %8s %9s %8s %9s %8s %9s %8s" % (
    "method","n","clipDir","editScr","insL1","bgL1","bgSSIM","bgLPIPS","bgDINO","lpipsF","srcL1","srcSSIM")
print(hdr); print("-"*len(hdr))
for r in summary:
    print("%-16s %4d %8.4f %9.4f %8.4f %8.4f %9.4f %8.4f %9.4f %8.4f %9.4f %8.4f" % (
        r["method"], r["n"], r["clip_direction_similarity"], r["edit_score"], r["inside_l1"],
        r["bg_l1"], r["bg_ssim_luma"], r["bg_lpips"], r["bg_dino_source"], r["lpips_full"],
        r["source_l1"], r["source_ssim_luma"]))
