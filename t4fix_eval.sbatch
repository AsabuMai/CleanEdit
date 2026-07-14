#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -J t4fix-eval
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/t4fix_eval_%j.log
set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/t4fix_eval
MASK_DIR=$PROJ/data/flowedit_compatible_135/eval_masks
LOCAL_PROMPTS=$PROJ/data/flowedit_compatible_135/local_target_prompts.json

export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LOCAL_PROMPTS_JSON=$LOCAL_PROMPTS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname; date
cd "$PROJ"
if [[ "$(hostname)" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then echo "Refusing outside a100-01" >&2; exit 2; fi

mkdir -p "$EXP"
"$PROJ/.venv/bin/python" prepare_flowedit135_metric_runs.py \
  --manifest "$PROJ/tmp/t4_eval_manifest.json" \
  --out-dir "$PROJ/outputs/t4fix_eval_runs" \
  --methods "ours_sd3 ours_flux ours_flux_t4fix"

"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJ/outputs/t4fix_eval_runs" \
  --csv-output "$EXP/metrics.csv" \
  --json-output "$EXP/metrics.json" \
  --task-names all \
  --method-names all \
  --seeds 10 \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --eval-mask-dir "$MASK_DIR"

"$PROJ/.venv/bin/python" - <<'PY'
import csv
from collections import defaultdict
rows=list(csv.DictReader(open("experiments/t4fix_eval/metrics.csv",newline="",encoding="utf-8")))
g=defaultdict(list)
for r in rows: g[r["method"]].append(r)
def m(items,k):
    v=[float(r[k]) for r in items if r.get(k) not in (None,"")]
    return sum(v)/len(v) if v else float("nan")
keys=["local_clip_t","clip_direction_similarity","clip_target_minus_source","bg_lpips","bg_l1","bg_dino_source","bg_ssim_luma"]
print("method            n  "+ "  ".join(f"{k:>22}" for k in keys))
for meth in ["ours_flux","ours_flux_t4fix","ours_sd3"]:
    if meth not in g: continue
    it=g[meth]
    print(f"{meth:16} {len(it):2}  " + "  ".join(f"{m(it,k):22.4f}" for k in keys))
PY
date
echo t4fix_eval_complete
