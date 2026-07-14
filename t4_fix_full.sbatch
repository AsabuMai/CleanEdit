#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:nvidia_h100_pcie:6
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH -J t4fix-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/t4fix_flux_%j.log

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

export MANIFEST=$PROJ/tmp/t4_full_manifest.json
export FLUX_OUT=outputs/fe135_t4fix_dece_flux
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export SEED=10

cd "$PROJ"
hostname; date
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')
echo "[gpu] $GPU_NAME"
if [[ "$GPU_NAME" != "NVIDIA H100 PCIe" ]]; then echo "[gpu-guard] got $GPU_NAME"; exit 75; fi

"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

echo "[count] $(find "$PROJ/$FLUX_OUT" -name result.png | wc -l)/19"
date
echo t4fix_flux_complete
