#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH -J fe135-flux-cat-tune
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_flux_tune_cat_sweep_pcie8_h100_%j.log

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
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_N_MAX=24
export SEED=10
export T5_POSE_PROMPT=1
export T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="text, letters, logo, watermark, fur, animal fur, whiskers, hair strands, fuzzy outline, extra limbs, pose change"

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

run_case() {
  echo
  echo "[case] $FLUX_OUT"
  date
  "$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v10.py
  find "$PROJ/$FLUX_OUT" -type f -name '*.png' | wc -l
}

export FLUX_OUT=outputs/fe135_flux_tune_cat_s12_preserve_v1
export FLUX_STEPS=12
export T5_EXPAND_EDIT_MASK=1
export T5_EXPAND_EDIT_RADIUS=9
export T5_EXPAND_EDIT_MASK_BLUR=3
export KIND_OVERRIDE_JSON='{"material":{"edit_hedit":"0.96","edit_region":"0.18","edit_target":"0.14","local_target":"0.58","rec":"0.36","struct":"0.72","traj":"0.34","pbud":"0.08","pgain":"1.8","outside_lock":"0.03"}}'
export ADAPT_OVERRIDE_JSON='{"material":{"--edit-local-target-prompt":"a carved wooden cat sculpture, same sitting pose, same silhouette","--edit-local-target-guidance-scale":"0.58","--edit-local-target-cfg-scale":"3.8","--tar-guidance-scale":"4.0","--edit-source-guidance-scale":"0.08","--region-target-outside-lock-scale":"0.03","--final-mask-blend-scale":"0.85","--final-mask-alpha-gamma":"1.4","--negative-prompt":"text, letters, logo, watermark, fur, whiskers, fuzzy outline, extra limbs, pose change"}}'
run_case

export FLUX_OUT=outputs/fe135_flux_tune_cat_s16_blend075_v2
export FLUX_STEPS=16
export T5_EXPAND_EDIT_MASK=1
export T5_EXPAND_EDIT_RADIUS=9
export T5_EXPAND_EDIT_MASK_BLUR=3
export KIND_OVERRIDE_JSON='{"material":{"edit_hedit":"0.92","edit_region":"0.16","edit_target":"0.12","local_target":"0.52","rec":"0.42","struct":"0.78","traj":"0.38","pbud":"0.07","pgain":"1.6","outside_lock":"0.04"}}'
export ADAPT_OVERRIDE_JSON='{"material":{"--edit-local-target-prompt":"a carved wooden cat sculpture, same sitting pose, same silhouette","--edit-local-target-guidance-scale":"0.52","--edit-local-target-cfg-scale":"3.6","--tar-guidance-scale":"3.8","--edit-source-guidance-scale":"0.10","--region-target-outside-lock-scale":"0.04","--final-mask-blend-scale":"0.75","--final-mask-alpha-gamma":"1.5","--negative-prompt":"text, letters, logo, watermark, fur, whiskers, fuzzy outline, extra limbs, pose change"}}'
run_case

export FLUX_OUT=outputs/fe135_flux_tune_cat_s16_noexpand_v3
export FLUX_STEPS=16
export T5_EXPAND_EDIT_MASK=0
export T5_EXPAND_EDIT_RADIUS=0
export T5_EXPAND_EDIT_MASK_BLUR=0
export KIND_OVERRIDE_JSON='{"material":{"edit_hedit":"0.88","edit_region":"0.15","edit_target":"0.10","local_target":"0.50","rec":"0.48","struct":"0.82","traj":"0.42","pbud":"0.06","pgain":"1.5","outside_lock":"0.05"}}'
export ADAPT_OVERRIDE_JSON='{"material":{"--edit-local-target-prompt":"a carved wooden cat sculpture, same sitting pose, same silhouette","--edit-local-target-guidance-scale":"0.50","--edit-local-target-cfg-scale":"3.5","--tar-guidance-scale":"3.6","--edit-source-guidance-scale":"0.12","--region-target-outside-lock-scale":"0.05","--final-mask-blend-scale":"0.70","--final-mask-alpha-gamma":"1.6","--negative-prompt":"text, letters, logo, watermark, fur, whiskers, fuzzy outline, extra limbs, pose change"}}'
run_case

date
echo fe135_flux_tune_cat_sweep_pcie8_h100_complete
