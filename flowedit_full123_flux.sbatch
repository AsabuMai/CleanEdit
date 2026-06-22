#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH -J fe123-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit_full123_flux_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
date
MANIFEST=$PROJ/data/flowedit_compatible_123/manifest.json \
FLUX_OUT=outputs/flowedit_full123_flux_20260619 \
SEED=10 \
"$PROJ/.venv-h100/bin/python" scripts/flowedit_flux_batch.py
date
