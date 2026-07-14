#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH -J s115-eval
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_s115_eval_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/fe135_t4boost_eval
MASK_DIR=$PROJ/data/flowedit_compatible_135/eval_masks
export HF_HOME=$PROJ/.cache/huggingface HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_PROMPTS_JSON=$PROJ/data/flowedit_compatible_135/local_target_prompts.json
hostname; date; cd "$PROJ"
mkdir -p "$EXP"
"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJ/outputs/flowedit135_metric_runs" \
  --csv-output "$EXP/metrics.csv" \
  --json-output "$EXP/metrics.json" \
  --task-names all \
  --method-names ours_flux_t4boost \
  --seeds 10 \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --eval-mask-dir "$MASK_DIR"
echo "[done] $(wc -l < "$EXP/metrics.csv") lines"; date; echo s115_eval_complete
