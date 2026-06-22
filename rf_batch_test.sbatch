#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J rf-batch-test
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/rf_batch_test_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DISABLE_BASELINE_NSFW=1
hostname; date
cd "$PROJ/_baselines/src/RF-Solver-Edit/FLUX_Image_Edit/src"
MANIFEST=$PROJ/data/pie_dev_20260618/manifest.json LIMIT=2 OUT=outputs/pie_rf_batch_test_20260618 \
  "$PROJ/_baselines/envs/rf-solver-edit-py310/bin/python" rfsolver_batch.py
date; echo rf_batch_test_complete
