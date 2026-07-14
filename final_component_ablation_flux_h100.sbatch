#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=04:30:00
#SBATCH -J compabl-flux
#SBATCH --array=0-2%1
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/final_component_ablation_flux_%A_%a.log

set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
VARS=(preserve adaptive opsupport)
VARIANT=${VARS[$SLURM_ARRAY_TASK_ID]}

export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_ablation_non_t4_116.json
export FLUX_OUT=outputs/final_component_ablation_non_t4_flux_${VARIANT}
if [[ "$VARIANT" == "opsupport" ]]; then
  export ABLATE=""
  export ADAPT_OVERRIDE_JSON='{"add":{"--support-candidate":"attention_x_clean"},"decal":{"--support-candidate":"attention_x_clean"},"material":{"--support-candidate":"attention_x_clean"}}'
else
  export ABLATE=$VARIANT
fi
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export SEED=10
export T5_EXPAND_EDIT_MASK=1
export T5_EXPAND_EDIT_RADIUS=21
export T5_EXPAND_EDIT_MASK_BLUR=6
export T5_POSE_PROMPT=1
export T5_AUTO_PROMPT_PREFIX=1
export T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark"

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

OUTDIR="$PROJ/$FLUX_OUT"
case "$OUTDIR" in
  "$PROJ"/outputs/final_component_ablation_non_t4_flux_*) rm -rf "$OUTDIR" ;;
  *) echo "refusing to clean $OUTDIR" >&2; exit 3 ;;
esac

"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

echo "[count]"
find "$PROJ/$FLUX_OUT" -path "*/seed_${SEED}/result.png" | wc -l
date
echo final_component_ablation_flux_${VARIANT}_complete
