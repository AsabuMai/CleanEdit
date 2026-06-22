#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=25:00:00
#SBATCH -J fe135-samflow-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit135_samflow_sd3_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ"
MODE=sd3 \
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest.json \
OUT=outputs/flowedit135_baselines/sam_flow_sd3 \
SEED=10 \
"$PROJ/_baselines/envs/sam-flow-py310/bin/python" scripts/flowedit_samflow_batch.py
date; echo flowedit135_samflow_sd3_complete
