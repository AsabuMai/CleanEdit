#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH -J fe118-t4mask-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe118_t4mask_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
cd "$PROJ"
echo "[1/2] generating SAM masks for T4"
"$PROJ/.venv/bin/python" scripts/make_fe118_t4_masks.py
echo "[2/2] CleanEdit SD3 with masks"
MANIFEST=$PROJ/data/flowedit_compatible_118/manifest_smoke_t4_mask.json OUT=outputs/fe118_smoke_t4mask_sd3 "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
date; echo fe118_t4mask_sd3_complete
