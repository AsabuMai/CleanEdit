#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J b6r2-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket6_r2_flux_%j.log

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

# Bucket 6 round 2: raw SAM masks are clean silhouettes; the background
# repaint happens INSIDE the r21/b6 expanded blob (FLUX freed to erase rock
# above the head / invent a sun in the castle sky). Shrink expansion to a
# shape-following ring (r9/b4), keep the round-1 outside-preserve boost.
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket6_r2.json
export FLUX_OUT=outputs/bucket6_r2_flux
export ABLATE=no_final_restore
export SEED=10
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export T5_EXPAND_EDIT_MASK=1
export T5_EXPAND_EDIT_RADIUS=9
export T5_EXPAND_EDIT_MASK_BLUR=4
export T5_POSE_PROMPT=1
export T5_AUTO_PROMPT_PREFIX=1
export T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark"
export KIND_OVERRIDE_JSON='{"t5_material":{"rec":0.40,"traj":0.30,"outside_lock":0.08}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
find "$PROJ/outputs/bucket6_r2_flux" -path "*/seed_10/result.png" | wc -l
date
echo bucket6_r2_complete
