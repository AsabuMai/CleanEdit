#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -J b46-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket46_flux_%j.log

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

# Buckets 4+6 joint FLUX experiment.
# Bucket 4 (NOED): pareto sweep proved insertions fire at 1.15x with no bg
# cost — raise t2_insert edit budget; t4_recolor hedit 0.55 -> 0.80 for the
# horse/sails no-edits.
# Bucket 6 (BGREP): t5_material outside preserve was near-zero (rec 0.15,
# traj 0.10, outside_lock 0.02) — raise it; sweep g070 shows the material
# transform quality survives lower effective edit, so this should be
# affordable.
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket46_flux.json
export FLUX_OUT=outputs/bucket46_flux
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
export KIND_OVERRIDE_JSON='{"t2_insert":{"edit_hedit":1.20,"local_target":1.10},"t4_recolor":{"edit_hedit":0.80,"local_target":0.35},"t5_material":{"rec":0.40,"traj":0.30,"outside_lock":0.08}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
find "$PROJ/outputs/bucket46_flux" -path "*/seed_10/result.png" | wc -l
date
echo bucket46_flux_complete
