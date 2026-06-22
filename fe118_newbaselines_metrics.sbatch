#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J fe118-newbl-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe118_newbaselines_metrics_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/flowedit118_newbaselines_fixedmask_metrics_20260620
MASK_DIR=$PROJ/data/flowedit_compatible_118/eval_masks
export HF_HOME=$PROJ/.cache/huggingface HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ"
mkdir -p "$EXP"
echo "[1/3] prepare new-baseline metric runs (OT-RF link, DRFS crop-right)"
"$PROJ/.venv/bin/python" prepare_newbaselines_metric_runs.py
echo "[2/3] evaluate paper metrics on new baselines"
"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJ/outputs/fe118_newbaselines_metric_runs" \
  --csv-output "$EXP/metrics.csv" --json-output "$EXP/metrics.json" \
  --task-names all --method-names all --seeds 10 \
  --clip-model openai/clip-vit-large-patch14 --dino-model facebook/dinov2-base \
  --compute-lpips --bg-dilate-px 31 --eval-mask-dir "$MASK_DIR"
echo "[3/3] FULL COMPARISON (all methods incl OT-RF + DRFS)"
"$PROJ/.venv/bin/python" build_fe118_final_compare.py
date; echo fe118_newbaselines_metrics_complete
