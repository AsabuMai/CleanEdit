#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J b6r5-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket6_r5_flux_%j.log

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

# Bucket 6 round 4: the r3 residual ghosts (fe_172/175 carved face above
# head, fe_146 sun blob) all sit OUTSIDE the mask. Default t5_material has
# outside_lock=0.02 — under no_final_restore the outside is nearly free.
# Keep expansion off, raise outside_lock hard; per-case hedit boost for
# fe_130's incomplete gilding.
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket6_r3.json
export FLUX_OUT=outputs/bucket6_r5_flux
export ABLATE=no_final_restore
export SEED=10
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export T5_EXPAND_EDIT_MASK=0
export T5_POSE_PROMPT=1
export T5_AUTO_PROMPT_PREFIX=1
export T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark, duplicated figure, second person, oversized figure"
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45}}'
export KEY_OVERRIDE_JSON='{"fe_130":{"--edit-hedit-guidance-scale":"1.55"}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
find "$PROJ/outputs/bucket6_r5_flux" -path "*/seed_10/result.png" | wc -l
date
echo bucket6_r4_complete
