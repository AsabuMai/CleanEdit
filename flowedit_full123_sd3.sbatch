#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH -J fe123-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit_full123_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
date
MANIFEST=$PROJ/data/flowedit_compatible_123/manifest.json \
OUT=outputs/flowedit_full123_sd3_20260619 \
SEED=10 \
"$PROJ/.venv/bin/python" scripts/flowedit_sd3_batch.py
date
