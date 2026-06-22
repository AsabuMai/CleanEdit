#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH -J fe123-reflex
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit123_reflex_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
date
cd "$PROJ"
MANIFEST=$PROJ/data/flowedit_compatible_123/manifest.json \
OUT=outputs/flowedit_baselines/reflex \
SEED=10 \
"$PROJ/_baselines/envs/reflex-py310/bin/python" scripts/flowedit_reflex_batch.py
date
