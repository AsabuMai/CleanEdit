#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J b9-resid-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket9_resid_flux_%j.log

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
cd "$PROJ"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"

export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket9_resid.json
test -f "$MANIFEST"
export FLUX_OUT=outputs/bucket9_resid_flux
export ABLATE=no_final_restore
export SEED=10
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export T5_EXPAND_EDIT_MASK=1
export T5_EXPAND_EDIT_RADIUS=21
export T5_EXPAND_EDIT_MASK_BLUR=6
export T5_POSE_PROMPT=1
export T5_AUTO_PROMPT_PREFIX=1
export T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark"
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
find "$PROJ/outputs/bucket9_resid_flux" -path "*/seed_10/result.png" | wc -l
date
echo bucket9_resid_flux_complete
