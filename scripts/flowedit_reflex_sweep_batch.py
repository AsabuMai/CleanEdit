from __future__ import annotations

import copy
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
SRC = PROJ / "_baselines/src/ReFlex"
MANIFEST = Path(os.environ.get("MANIFEST", str(PROJ / "data/flowedit_compatible_135/manifest_pareto_non_t4_33.json")))
OUT = PROJ / os.environ.get("OUT", "outputs/pareto_sweep_20260701/reflex_g100")
METHOD = os.environ.get("METHOD", "reflex")
LIMIT = int(os.environ.get("LIMIT", "0"))
SEED = int(os.environ.get("SEED", "10"))
GUIDANCE = os.environ.get("REFLEX_GUIDANCE", "3.5")
FEATURE_STEPS = os.environ.get("REFLEX_FEATURE_STEPS", "5")
ATTN_TOPK = os.environ.get("REFLEX_ATTN_TOPK", "5")

os.chdir(str(SRC))
sys.path.insert(0, str(SRC))
man = json.load(MANIFEST.open())
if LIMIT:
    man = man[:LIMIT]

e0 = man[0]
sys.argv = [
    "img_edit.py",
    "--gpu",
    "0",
    "--seed",
    str(SEED),
    "--img_path",
    str(Path(e0["image"]).resolve()),
    "--source_prompt",
    e0["source_prompt"],
    "--target_prompt",
    e0["target_prompt"],
    "--results_dir",
    str(OUT / "_tmp"),
    "--feature_steps",
    FEATURE_STEPS,
    "--attn_topk",
    ATTN_TOPK,
    "--guidance_scale",
    GUIDANCE,
]
import img_edit  # noqa: E402


_orig = img_edit.get_flux_pipeline
_cache = {}


def _get_pipe(*args, **kwargs):
    if "pipe" not in _cache:
        _cache["pipe"] = _orig(*args, **kwargs)
        print("[pareto-reflex] pipe loaded once", flush=True)
    return _cache["pipe"]


img_edit.get_flux_pipeline = _get_pipe
base = img_edit.args
done = 0
for item in man:
    final = OUT / item["key"] / METHOD / f"seed_{SEED}" / "result.png"
    if final.exists():
        print("skip", item["key"], flush=True)
        continue
    final.parent.mkdir(parents=True, exist_ok=True)
    raw = final.parent / "raw"
    args = copy.deepcopy(base)
    args.img_path = str(Path(item["image"]).resolve())
    args.source_prompt = item["source_prompt"]
    args.target_prompt = item["target_prompt"]
    args.results_dir = str(raw)
    args.seed = SEED
    args.guidance_scale = float(GUIDANCE)
    args.feature_steps = int(FEATURE_STEPS)
    args.attn_topk = int(ATTN_TOPK)
    t0 = time.time()
    try:
        img_edit.main(args)
        found = glob.glob(str(raw / "**" / "target_0.png"), recursive=True)
        if not found:
            print("FAILED-nooutput", item["key"], flush=True)
            continue
        shutil.copy(found[0], final)
        meta = {
            "method": METHOD,
            "seed": SEED,
            "reflex_guidance": float(GUIDANCE),
            "feature_steps": int(FEATURE_STEPS),
            "attn_topk": int(ATTN_TOPK),
            "source_prompt": item["source_prompt"],
            "target_prompt": item["target_prompt"],
        }
        (final.parent / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        done += 1
        print("OK", item["key"], "%.1fs" % (time.time() - t0), flush=True)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print("FAILED", item["key"], repr(exc), flush=True)
print("ALLDONE pareto-reflex", done, "/", len(man), flush=True)
