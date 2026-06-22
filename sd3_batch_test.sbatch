#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J sd3-batch-test
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/sd3_batch_test_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
cd "$PROJ"
LIMIT=2 OUT=outputs/pie_sd3_batch_test_20260618 "$PROJ/.venv-h100/bin/python" scripts/pie_sd3_batch.py
date; echo sd3_batch_test_complete
