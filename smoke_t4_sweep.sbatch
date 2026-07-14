#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:nvidia_h100_pcie:6
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH -J smoke-t4-sweep
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/smoke_t4_sweep_%j.log

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

export MANIFEST=$PROJ/tmp/smoke_t4_recolor_manifest.json
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

run() {
  local out=$1; shift
  export FLUX_OUT="outputs/$out"
  echo "=== RUN $out :: $* ==="
  env "$@" "$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
}

# Variant A: weaken flat prompt-edit, let textured projection carry the color
run smoke_t4_vA KIND_OVERRIDE_JSON='{"recolor":{"edit_hedit":0.18}}' RC_PROJ=0.55 RC_MINIMAL=0.30 RC_TEXPRES=1.0 RC_LUMA=0.80 RC_ALPHA=0.85

# Variant B: desaturate + lower alpha, moderate projection
run smoke_t4_vB KIND_OVERRIDE_JSON='{"recolor":{"edit_hedit":0.28}}' RC_PROJ=0.30 RC_MINIMAL=0.18 RC_TEXPRES=1.0 RC_LUMA=0.85 RC_ALPHA=0.60

echo "[count] vA=$(find outputs/smoke_t4_vA -name result.png|wc -l) vB=$(find outputs/smoke_t4_vB -name result.png|wc -l)"
date
echo smoke_t4_sweep_complete
