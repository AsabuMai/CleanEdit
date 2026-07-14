#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH -J flux-t4boost
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_t4boost_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub TRANSFORMERS_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MANIFEST=$PROJ/tmp/t4_manifest.json
export FLUX_OUT=outputs/fe135_flux_t4boost
# boost recolor color guidance + a bit of edit strength; DO NOT reopen RC projection (RC_*=0 defaults)
export KIND_OVERRIDE_JSON='{"t4_recolor":{"color":"0.45","edit_hedit":0.72,"edit_target":0.10}}'
export FLUX_LOCAL_FILES_ONLY=1 FLUX_OFFLOAD=model FLUX_STEPS=12 FLUX_N_MAX=10 SEED=10
export T5_EXPAND_EDIT_MASK=1 T5_EXPAND_EDIT_RADIUS=21 T5_EXPAND_EDIT_MASK_BLUR=6 T5_POSE_PROMPT=1 T5_AUTO_PROMPT_PREFIX=1 T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark"
hostname; date; cd "$PROJ"
if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then echo "wrong node">&2; exit 2; fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 | sed 's/^ *//;s/ *$//')
if [[ "$GPU_NAME" != "NVIDIA H100 PCIe" ]]; then echo "[gpu-guard] got $GPU_NAME"; exit 75; fi
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
echo "[count] $(find $PROJ/$FLUX_OUT -name result.png|wc -l)/19"; date; echo flux_t4boost_complete
