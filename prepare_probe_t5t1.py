import json, os
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MAN = json.load(open(PROJ / "data/flowedit_compatible_118/manifest_probe_t5t1.json"))
OUT = PROJ / "outputs/probe_t5t1_metric_runs"
UNIQ = PROJ / "data/flowedit_compatible_118/unique_inputs"

SRC = {
    "dece_current": PROJ / "outputs/fe118_r3full_dece_sd3/{k}/support_v3_controller_rmsgap/seed_10/result.png",
    "dece_A":       PROJ / "outputs/probe_t5t1_sd3_A/{k}/support_v3_controller_rmsgap/seed_10/result.png",
    "dece_B":       PROJ / "outputs/probe_t5t1_sd3_B/{k}/support_v3_controller_rmsgap/seed_10/result.png",
    "sam_flow_sd3": PROJ / "outputs/flowedit118_metric_runs/{k}/sam_flow_sd3/seed_10/result.png",
}

def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        from shutil import copy2; copy2(src, dst)

counts = {m: 0 for m in SRC}
missing = []
for e in MAN:
    k = e["key"]
    for method, tmpl in SRC.items():
        sp = Path(str(tmpl).format(k=k))
        if not sp.exists():
            missing.append((k, method)); continue
        rd = OUT / k / method / "seed_10"
        rd.mkdir(parents=True, exist_ok=True)
        link(sp, rd / "result.png")
        meta = {"method": method, "task": k, "seed": 10,
                "image": str(UNIQ / (k + ".png")), "source_image": str(UNIQ / (k + ".png")),
                "source_prompt": e["source_prompt"], "target_prompt": e["target_prompt"],
                "family": e.get("family"), "family_label": e.get("family_label")}
        (rd / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (rd / "stats.json").write_text(json.dumps({"steps": []}) + "\n", encoding="utf-8")
        (rd / "command.txt").write_text(method + " probe link " + k + "\n", encoding="utf-8")
        counts[method] += 1
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
print("counts", counts, "missing", len(missing), missing)
