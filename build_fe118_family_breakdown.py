import csv, json
from collections import defaultdict
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
man = {e["key"]: e for e in json.load(open(PROJ / "data/flowedit_compatible_118/manifest.json"))}
FAMS = ["T1", "T2", "T3", "T4", "T5"]
def fam_of(task):
    fl = (man.get(task, {}).get("family_label") or "")
    return fl[:2] if fl[:2] in FAMS else "?"
SRCS = [PROJ / "experiments/flowedit118_baselines_fixedmask_metrics_20260619/metrics.csv",
        PROJ / "experiments/flowedit118_dece_fixedmask_metrics_20260619/metrics.csv"]
def fnum(r, k):
    try: return float(r.get(k, "") or 0.0)
    except ValueError: return 0.0
cell = defaultdict(lambda: defaultdict(list))   # metric -> (method,fam) -> values
methods = set()
for src in SRCS:
    if not src.exists(): print("MISSING", src); continue
    for r in csv.DictReader(src.open(newline="", encoding="utf-8")):
        m = r["method"]; f = fam_of(r["task"]); methods.add(m)
        for met in ("edit_score", "bg_l1", "inside_l1", "clip_direction_similarity"):
            cell[met][(m, f)].append(fnum(r, met))
order = [m for m in ["fireflow","rf_solver_edit","flowedit_flux","flowedit_sd3","splitflow_sd3","reflex","sam_flow_flux","sam_flow_sd3","dece_rf_sd3","dece_rf_flux"] if m in methods]
counts = {f: sum(1 for k,e in man.items() if (e.get("family_label") or "")[:2]==f) for f in FAMS}
def show(met, title):
    print("\n=== %s  (per-family mean) ===" % title)
    print("%-16s %s" % ("method", " ".join("%8s" % (f+"(%d)"%counts[f]) for f in FAMS)))
    for m in order:
        vals=[]
        for f in FAMS:
            v=cell[met][(m,f)]; vals.append("%8.4f" % (sum(v)/len(v)) if v else "       -")
        print("%-16s %s" % (m, " ".join(vals)))
show("edit_score", "editScr (edit magnitude, higher=more edit)")
show("inside_l1", "inside_L1 (edit-region change, higher=more edit)")
show("bg_l1", "bgL1 (background change, lower=better preserve)")
