import csv, json
from collections import defaultdict
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
man = {e["key"]: e for e in json.load(open(PROJ / "data/flowedit_compatible_118/manifest_probe2.json"))}
PROBE = set(man)
def fam(k): return (man.get(k, {}).get("family_label") or "?")[:2]
ORIG = PROJ / "experiments/flowedit118_dece_fixedmask_metrics_20260619/metrics.csv"
R3 = PROJ / "experiments/flowedit118_r3_fixedmask_metrics_20260619/metrics.csv"
def load(p, method):
    d = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        if r["task"] not in PROBE or r["method"] != method: continue
        for m in ("edit_score", "inside_l1", "bg_l1"):
            try: v = float(r.get(m, "") or 0.0)
            except ValueError: v = 0.0
            d[fam(r["task"])][m].append(v)
    return d
for method in ("dece_rf_sd3", "dece_rf_flux"):
    o = load(ORIG, method); b = load(R3, method)
    print("=== %s : editScr & bgL1  orig -> r3 ===" % method)
    for f in ["T1", "T2", "T4", "T5"]:
        ov = o[f]["edit_score"]; bv = b[f]["edit_score"]; obg = o[f]["bg_l1"]; bbg = b[f]["bg_l1"]
        if not ov and not bv: continue
        mo = sum(ov)/len(ov) if ov else 0; mb = sum(bv)/len(bv) if bv else 0
        go = sum(obg)/len(obg) if obg else 0; gb = sum(bbg)/len(bbg) if bbg else 0
        ctl = " (control)" if f == "T5" else ""
        print("  %s editScr %.4f -> %.4f (%+.4f)   bgL1 %.4f -> %.4f (%+.4f)%s" % (f, mo, mb, mb-mo, go, gb, gb-go, ctl))
