#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH -J fe135-flux-v10-direct
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_flux_v10_smoke5_direct_h100_%j.log

set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_flux_v10_smoke5.json
export FLUX_OUT=outputs/fe135_flux_v10_smoke5_direct_h100
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=none
export SEED=10
export T5_EXPAND_EDIT_MASK=1
export T5_EXPAND_EDIT_RADIUS=21
export T5_EXPAND_EDIT_MASK_BLUR=6
export T5_PROMPT_PREFIX="A smooth clean-edged wooden cat sculpture with no fur, no whiskers, and no floating hair strands"
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines"

hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside h100-01" >&2
  exit 2
fi

"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v10.py

date
echo fe135_flux_v10_smoke5_direct_h100_complete
