#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=06:00:00
#SBATCH -J compabl-met
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/final_component_ablation_metrics_%j.log

set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/final_component_ablation_non_t4_20260630
RUNS=$PROJ/outputs/final_component_ablation_runs_20260630
MASK_DIR=$PROJ/data/flowedit_compatible_135/eval_masks
LOCAL_PROMPTS=$PROJ/data/flowedit_compatible_135/local_target_prompts.json

export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LOCAL_PROMPTS_JSON=$LOCAL_PROMPTS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

mkdir -p "$EXP"
"$PROJ/.venv/bin/python" scripts/prepare_final_component_ablation_runs.py

"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$RUNS" \
  --csv-output "$EXP/metrics.csv" \
  --json-output "$EXP/metrics.json" \
  --task-names all \
  --method-names "full_sd3 sd3_wopreserve sd3_wofeedback sd3_woopsupport full_flux flux_wopreserve flux_wofeedback flux_woopsupport" \
  --seeds "10" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --eval-mask-dir "$MASK_DIR"

"$PROJ/.venv/bin/python" scripts/summarize_metrics_csv.py \
  --metrics "$EXP/metrics.csv" \
  --out-dir "$EXP" \
  --expected-per-method 116

date
echo final_component_ablation_metrics_complete
