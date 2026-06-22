#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH -J fe-sam-mask
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit_sam_masks_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface HF_HUB_OFFLINE=1
hostname; date
cd "$PROJ"
"$PROJ/.venv/bin/python" logs/make_flowedit_sam_masks.py
date; echo flowedit_sam_masks_complete
