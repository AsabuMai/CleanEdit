#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J ka3-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/ka3_flux_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
MANIFEST=$PROJ/data/pie_pilot_20260618/manifest_kindaware_3.json FLUX_OUT=outputs/pie_kindaware_flux_fixcheck_20260619 "$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware.py
date; echo ka3_flux_complete
