#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J fe135-t4-tex-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t4_texture_flux_%j.log

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

export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t4_recolor_19.json
export FLUX_OUT=outputs/fe135_t4_texture_flux_h100
export ABLATE=no_final_postprocess
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export SEED=10
export RC_MINIMAL=0.0
export RC_BLEND=0.0
export RC_PROJ=0.20
export RC_TEXPRES=0.85
export RC_ALPHA=0.85
export RC_LUMA=0.65

hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside h100-01" >&2
  exit 2
fi

"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

echo "[count]"
find "$PROJ/$FLUX_OUT" -path "*/seed_${SEED}/result.png" | wc -l
date
echo fe135_t4_texture_flux_complete
