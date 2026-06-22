#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH -J fe135-fix2-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_delta17_fix2_flux_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ"
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_sam_delta17_fix2.json \
FLUX_OUT=outputs/fe135_delta17_fix2_dece_flux \
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_kindonly.py
date; echo fe135_delta17_fix2_flux_complete
