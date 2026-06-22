import json, subprocess
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
src = PROJ / "data/flowedit_compatible_118/manifest_smoke_t4t5.json"
out_dir = PROJ / "data/flowedit_compatible_118/sam_masks_t4"
out_dir.mkdir(parents=True, exist_ok=True)
rows = [e for e in json.load(open(src)) if e.get("family_label", "").startswith("T4")]
enriched = []
for e in rows:
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
        subprocess.run(cmd, cwd=str(PROJ), check=True)
    ee = dict(e)
    ee["pp_local_mask"] = str(mask.relative_to(PROJ))
    ee["global_mask"] = False
    ee["sam_phrase"] = phrase
    enriched.append(ee)
dst = PROJ / "data/flowedit_compatible_118/manifest_smoke_t4_mask.json"
json.dump(enriched, open(dst, "w"), indent=2)
print("WROTE", dst)
for e in enriched:
    print(e["key"], e["sam_phrase"], e["pp_local_mask"])
