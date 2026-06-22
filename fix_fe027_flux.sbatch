#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH -J fix-fe027-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fix_fe027_flux_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ"
MANIFEST=$PROJ/data/flowedit_compatible_118/manifest_fe027.json FLUX_OUT=outputs/fe118_full_dece_flux "$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware.py
date; echo fix_fe027_flux_complete
