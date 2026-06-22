import csv
from collections import defaultdict
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
DECE = PROJ / "experiments/flowedit118_dece_fixedmask_metrics_20260619"
BASE = PROJ / "experiments/flowedit118_baselines_fixedmask_metrics_20260619/summary_by_method.csv"
COLS = ["method", "n", "clip_t", "clip_target_minus_source", "clip_direction_similarity",
        "clip_image_source_similarity", "dino_source_similarity", "lpips_full", "lpips_edit",
        "lpips_bg", "source_l1", "source_l1_edit", "source_l1_bg", "source_ssim_luma",
        "source_ssim_luma_edit", "source_ssim_luma_bg"]
def fnum(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0
rows = list(csv.DictReader((DECE / "metrics.csv").open(newline="", encoding="utf-8")))
groups = defaultdict(list)
for r in rows:
    groups[r["method"]].append(r)
dece_summary = []
for method, items in sorted(groups.items()):
    rec = {"method": method, "n": len(items)}
    for k in COLS[2:]:
        rec[k] = sum(fnum(r, k) for r in items) / max(1, len(items))
    dece_summary.append(rec)
with (DECE / "summary_by_method.csv").open("w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=COLS); w.writeheader(); w.writerows(dece_summary)
combined = []
if BASE.exists():
    for r in csv.DictReader(BASE.open(newline="", encoding="utf-8")):
        combined.append({k: r.get(k, "") for k in COLS})
combined += [{k: rec.get(k, "") for k in COLS} for rec in dece_summary]
out = DECE / "comparison_baseline_vs_dece.csv"
with out.open("w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=COLS); w.writeheader(); w.writerows(combined)
print("WROTE", out)
for rec in combined:
    print(rec["method"], "n="+str(rec["n"]),
          "clip_dir=%.4f" % float(rec["clip_direction_similarity"] or 0),
          "l1_bg=%.4f" % float(rec["source_l1_bg"] or 0),
          "ssim_bg=%.4f" % float(rec["source_ssim_luma_bg"] or 0))
