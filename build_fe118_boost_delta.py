import csv, json
from collections import defaultdict
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
man = {e["key"]: e for e in json.load(open(PROJ / "data/flowedit_compatible_118/manifest_probe.json"))}
PROBE = set(man)
def fam(k): return (man.get(k, {}).get("family_label") or "?")[:2]
ORIG = PROJ / "experiments/flowedit118_dece_fixedmask_metrics_20260619/metrics.csv"
BOOST = PROJ / "experiments/flowedit118_boost_fixedmask_metrics_20260619/metrics.csv"
def load(p):
    d = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        if r["task"] not in PROBE: continue
        for m in ("edit_score", "inside_l1", "bg_l1"):
            try: v = float(r.get(m, "") or 0.0)
            except ValueError: v = 0.0
            d[(r["method"], fam(r["task"]))][m].append(v)
    return d
o = load(ORIG); b = load(BOOST)
FAMS = ["T1", "T2", "T3", "T4", "T5"]
for method in ("dece_rf_sd3", "dece_rf_flux"):
    print("\n=== %s : editScr (orig -> boost)  [bgL1 orig -> boost] ===" % method)
    for f in FAMS:
        ov = o[(method, f)]["edit_score"]; bv = b[(method, f)]["edit_score"]
        obg = o[(method, f)]["bg_l1"]; bbg = b[(method, f)]["bg_l1"]
        if not ov and not bv: continue
        mo = sum(ov)/len(ov) if ov else 0; mb = sum(bv)/len(bv) if bv else 0
        go = sum(obg)/len(obg) if obg else 0; gb = sum(bbg)/len(bbg) if bbg else 0
        print("  %s  editScr %.4f -> %.4f  (%+.4f)   bgL1 %.4f -> %.4f  (%+.4f)" % (f, mo, mb, mb-mo, go, gb, gb-go))
