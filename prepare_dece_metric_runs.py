import json, os
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_118/manifest.json"
OUT = PROJ / "outputs/flowedit118_dece_metric_runs"
METHODS = ["dece_rf_sd3", "dece_rf_flux"]
def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        from shutil import copy2
        copy2(src, dst)
def result_path(item, method):
    key = item["key"]
    if method == "dece_rf_sd3":
        return PROJ / "outputs/fe118_full_dece_sd3" / key / "support_v3_controller_rmsgap/seed_10/result.png"
    if method == "dece_rf_flux":
        return PROJ / "outputs/fe118_full_dece_flux" / key / "dece_rf_flux/seed_10/result.png"
    return None
def write_run(item, method, src):
    run_dir = OUT / item["key"] / method / "seed_10"
    run_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(src, run_dir / "result.png")
    meta = {"method": method, "task": item["key"], "seed": 10,
            "image": item["image"], "source_image": item["image"],
            "source_prompt": item["source_prompt"], "target_prompt": item["target_prompt"],
            "family": item.get("family"), "family_label": item.get("family_label"),
            "flowedit_idx": item.get("flowedit_idx"), "original_result": str(src)}
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (run_dir / "stats.json").write_text(json.dumps({"steps": []}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text("dece-rf result linked from " + str(src) + "\n", encoding="utf-8")
def main():
    manifest = json.load(MANIFEST.open())
    counts = {m: 0 for m in METHODS}
    missing = []
    for item in manifest:
        for method in METHODS:
            p = result_path(item, method)
            if p is None or not p.exists():
                missing.append({"task": item["key"], "method": method, "path": str(p or "")})
                continue
            write_run(item, method, p)
            counts[method] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    print("out", OUT)
    print("counts", counts)
    print("missing", len(missing))
    if missing:
        raise SystemExit(1)
if __name__ == "__main__":
    main()
