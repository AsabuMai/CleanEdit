#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH -J fe-run-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe_run_flux_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ"
MANIFEST=${MANIFEST:-$PROJ/data/flowedit_compatible_118/manifest_118_run.json} \
FLUX_OUT=${FLUX_OUT:-outputs/fe118_full_dece_flux} \
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware.py
date; echo fe_run_flux_complete
