#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH -J fe135-sam17
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_sam17_%j.log

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
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_17new_uniq.json \
OUT_DIR=$PROJ/data/flowedit_compatible_135/sam_support_masks_17new \
OUT_MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_sam_17new.json \
"$PROJ/.venv/bin/python" scripts/make_flowedit118_sam_support_manifest.py
"$PROJ/.venv/bin/python" scripts/merge_flowedit135_sam_manifest.py
date
echo fe135_sam17_complete
