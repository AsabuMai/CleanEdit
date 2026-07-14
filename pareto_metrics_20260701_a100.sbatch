#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH -J pareto-met
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pareto_metrics_20260701_%j.log

set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/pareto_sweep_metrics_20260701
RUNS=$PROJ/outputs/pareto_metric_runs_20260701
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
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"
mkdir -p "$EXP"

python3 scripts/prepare_pareto_20260701_metric_runs.py
METHODS=$(cat "$RUNS/_methods.txt")

"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$RUNS" \
  --csv-output "$EXP/metrics.csv" \
  --json-output "$EXP/metrics.json" \
  --task-names all \
  --method-names "$METHODS" \
  --seeds "10" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --eval-mask-dir "$MASK_DIR"

"$PROJ/.venv/bin/python" scripts/summarize_metrics_csv.py \
  --metrics "$EXP/metrics.csv" \
  --out-dir "$EXP" \
  --expected-per-method 33

date
echo pareto_metrics_20260701_complete
