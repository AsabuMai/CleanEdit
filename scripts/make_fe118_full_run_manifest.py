import json, subprocess
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
src = PROJ / "data/flowedit_compatible_118/manifest.json"
out_dir = PROJ / "data/flowedit_compatible_118/sam_masks_t4"
out_dir.mkdir(parents=True, exist_ok=True)
rows = json.load(open(src))
out = []
n_mask = 0
for e in rows:
    ee = dict(e)
    if e.get("family_label", "").startswith("T4"):
        amap = e.get("pp_aspect_mapping") or {}
        phrase = next(iter(amap.keys()), None) or e["source_prompt"].split()[0]
        stem = e["key"] + "_" + phrase.replace(" ", "_") + "_sam"
        mask = out_dir / (stem + ".png")
        meta = out_dir / (stem + ".json")
        if not mask.exists():
            cmd = [str(PROJ / ".venv/bin/python"), "scripts/make_semantic_mask.py",
                   "--image", e["image"], "--source-prompt", e["source_prompt"],
                   "--prompt", e["target_prompt"], "--phrase", phrase,
                   "--output", str(mask), "--metadata-output", str(meta),
                   "--max-image-size", "512", "--device", "cuda:0",
                   "--max-box-area-ratio", "0.70", "--box-threshold", "0.18",
                   "--text-threshold", "0.15", "--dilate", "6", "--blur", "5",
                   "--mask-mode", "sam_box_intersect"]
            print("MASK", e["key"], phrase, flush=True)
            try:
                subprocess.run(cmd, cwd=str(PROJ), check=True)
            except subprocess.CalledProcessError as ex:
                print("MASKFAIL", e["key"], repr(ex), flush=True)
        if mask.exists():
            ee["pp_local_mask"] = str(mask.relative_to(PROJ))
            ee["global_mask"] = False
            ee["sam_phrase"] = phrase
            n_mask += 1
        else:
            print("FALLBACK-MASKFREE", e["key"], flush=True)
    out.append(ee)
dst = PROJ / "data/flowedit_compatible_118/manifest_118_run.json"
json.dump(out, open(dst, "w"), indent=1)
print("WROTE", dst, "total", len(out), "T4masked", n_mask, flush=True)
