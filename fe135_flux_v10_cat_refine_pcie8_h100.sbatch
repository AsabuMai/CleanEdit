#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH -J fe135-flux-cat-refine
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_flux_v10_cat_refine_pcie8_h100_%j.log

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

export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_flux_v10_cat_refine.json
export FLUX_OUT=outputs/fe135_flux_v10_cat_refine_pcie8_h100
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=28
export FLUX_N_MAX=24
export SEED=10
export T5_EXPAND_EDIT_MASK=1
export T5_EXPAND_EDIT_RADIUS=15
export T5_EXPAND_EDIT_MASK_BLUR=5
export T5_POSE_PROMPT=1
export T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines"
export KIND_OVERRIDE_JSON='{"material":{"edit_hedit":"1.02","edit_region":"0.22","edit_target":"0.18","local_target":"0.72","rec":"0.28","struct":"0.58","traj":"0.22","pbud":"0.10","pgain":"2.4","outside_lock":"0.01"}}'
export ADAPT_OVERRIDE_JSON='{"material":{"--edit-local-target-prompt":"a wooden cat sculpture with carved wood grain, sitting in the same pose, clean edges, no fur, no whiskers","--edit-local-target-guidance-scale":"0.72","--edit-local-target-cfg-scale":"4.5"}}'

hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside h100-01" >&2
  exit 2
fi

GPU_LINE=$(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader,nounits | head -1)
echo "[gpu] $GPU_LINE"
GPU_NAME=$(echo "$GPU_LINE" | awk -F, '{gsub(/^ +| +$/,"",$1); print $1}')
if [[ "$GPU_NAME" != "NVIDIA H100 PCIe" ]]; then
  echo "[gpu-guard] want H100 PCIe, got name=$GPU_NAME"
  exit 75
fi

"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v10.py

date
echo fe135_flux_v10_cat_refine_pcie8_h100_complete
