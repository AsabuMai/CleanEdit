#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH -J fe-maskgen
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe_maskgen_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
cd "$PROJ"
# generates T4 SAM masks + writes data/flowedit_compatible_118/manifest_118_run.json
"$PROJ/.venv/bin/python" scripts/make_fe118_full_run_manifest.py
date; echo fe_maskgen_complete
