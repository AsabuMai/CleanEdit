import json, subprocess, shutil
from pathlib import Path
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
KEY = "fe_027_butterflies_1_yellow"
PHRASE = "butterflies"
man = json.load(open(PROJ / "data/flowedit_compatible_118/manifest_118_run.json"))
e = next(x for x in man if x["key"] == KEY)
out_dir = PROJ / "data/flowedit_compatible_118/sam_masks_t4"
out_dir.mkdir(parents=True, exist_ok=True)
stem = KEY + "_" + PHRASE + "_sam"
mask = out_dir / (stem + ".png")
meta = out_dir / (stem + ".json")
cmd = [str(PROJ / ".venv/bin/python"), "scripts/make_semantic_mask.py",
       "--image", e["image"], "--source-prompt", e["source_prompt"],
       "--prompt", e["target_prompt"], "--phrase", PHRASE,
       "--output", str(mask), "--metadata-output", str(meta),
       "--max-image-size", "512", "--device", "cuda:0",
       "--max-box-area-ratio", "0.70", "--box-threshold", "0.15",
       "--text-threshold", "0.12", "--dilate", "6", "--blur", "5",
       "--mask-mode", "sam_box_intersect"]
print("MASK", KEY, PHRASE, flush=True)
subprocess.run(cmd, cwd=str(PROJ), check=True)
ee = dict(e)
ee["pp_local_mask"] = str(mask.relative_to(PROJ))
ee["global_mask"] = False
ee["sam_phrase"] = PHRASE
json.dump([ee], open(PROJ / "data/flowedit_compatible_118/manifest_fe027.json", "w"), indent=1)
# clear stale outputs so runners regenerate this case
for d in ["outputs/fe118_full_dece_sd3/" + KEY, "outputs/fe118_full_dece_flux/" + KEY]:
    p = PROJ / d
    if p.exists():
        shutil.rmtree(p); print("removed", d, flush=True)
print("FE027 FIX PREP DONE", flush=True)
