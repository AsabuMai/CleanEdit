#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH -J fix-fe027-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fix_fe027_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
cd "$PROJ"
"$PROJ/.venv/bin/python" scripts/fix_fe027_mask.py
MANIFEST=$PROJ/data/flowedit_compatible_118/manifest_fe027.json OUT=outputs/fe118_full_dece_sd3 "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
date; echo fix_fe027_sd3_complete
