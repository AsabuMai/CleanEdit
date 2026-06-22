#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH -J t4-mask-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/t4_mask_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
MANIFEST=$PROJ/data/pie_pilot_20260618/manifest_pp_t4mask.json OUT=outputs/pie_kindaware_sd3_t4localmask_20260619 "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
date; echo t4_mask_sd3_complete
