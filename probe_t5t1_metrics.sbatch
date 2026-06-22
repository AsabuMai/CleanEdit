#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH -J probe-t5t1-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/probe_t5t1_metrics_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/probe_t5t1_metrics
MASK_DIR=$PROJ/data/flowedit_compatible_118/eval_masks
LP=$PROJ/data/flowedit_compatible_135/local_target_prompts.json
export HF_HOME=$PROJ/.cache/huggingface HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_PROMPTS_JSON=$LP
hostname; date
cd "$PROJ"
mkdir -p "$EXP"
echo "[1/3] prepare probe metric runs"
"$PROJ/.venv/bin/python" prepare_probe_t5t1.py
echo "[2/3] evaluate paper metrics (with focused local prompts)"
"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJ/outputs/probe_t5t1_metric_runs" \
  --csv-output "$EXP/metrics.csv" --json-output "$EXP/metrics.json" \
  --task-names all --method-names all --seeds 10 \
  --clip-model openai/clip-vit-large-patch14 --dino-model facebook/dinov2-base \
  --compute-lpips --bg-dilate-px 31 --eval-mask-dir "$MASK_DIR"
echo "[3/3] summary"
"$PROJ/.venv/bin/python" summarize_probe_t5t1.py
date; echo probe_t5t1_metrics_complete
