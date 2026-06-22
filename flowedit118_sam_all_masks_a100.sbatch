#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -J fe118-sam-all
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe118_sam_all_%j.log

set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
cd "$PROJ" || exit 1
"$PROJ/.venv/bin/python" scripts/make_flowedit118_sam_support_manifest.py
date
echo fe118_sam_all_complete
