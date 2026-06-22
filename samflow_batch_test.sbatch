#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:50:00
#SBATCH -J sf-batch-test
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/sf_batch_test_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DISABLE_BASELINE_NSFW=1
hostname; date
MODE=flux LIMIT=2 OUT=outputs/pie_sf_batch_test_20260618 "$PROJ/_baselines/envs/sam-flow-py310/bin/python" "$PROJ/scripts/sam_flow_batch.py"
date; echo sf_batch_test_complete
