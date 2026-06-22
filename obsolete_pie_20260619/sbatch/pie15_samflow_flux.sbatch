#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH -J pie15-sf-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie15_samflow_flux_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 DISABLE_BASELINE_NSFW=1
hostname; date
MODE=flux OUT=outputs/pie15_baselines_20260618 "$PROJ/.venv/bin/python" scripts/pie_samflow_run.py
date; echo pie15_samflow_flux_complete
