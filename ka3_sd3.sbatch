#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J ka3-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/ka3_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
MANIFEST=$PROJ/data/pie_pilot_20260618/manifest_kindaware_3.json OUT=outputs/pie_kindaware_sd3_fixcheck_20260619 "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
date; echo ka3_sd3_complete
