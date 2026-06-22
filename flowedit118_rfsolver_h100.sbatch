#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH -J fe118-rfsolver
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit118_rfsolver_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
date
cd "$PROJ"
"$PROJ/_baselines/envs/rf-solver-edit-py310/bin/python" scripts/run_flux_persistent_baselines.py \
  --manifest data/flowedit_compatible_118/baseline_rf_solver_edit.csv \
  --baselines rf_solver_edit \
  --device 0 \
  --max-image-size 512 \
  --skip-complete
date
