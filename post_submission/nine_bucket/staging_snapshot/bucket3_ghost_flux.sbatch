#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J b3-ghost
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket3_ghost_flux_%j.log

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

# Bucket 3 (FLUX GHOST): FLUX writes new glyphs without erasing the old ones.
# The port runs true-CFG, so a real negative prompt is available; derive the
# removed word from the source/target prompt diff (T3_NEG_FROM_REMOVED) and
# push "the word STOP / ghost letters / double exposure" negative.
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket23_t3.json
export ABLATE=no_final_restore
export SEED=10
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export T3_NEG_FROM_REMOVED=1

# Variant A: negative prompt only.
export FLUX_OUT=outputs/bucket3_neg_flux
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

# Variant B: negative prompt + stronger source suppression / erase budget.
export FLUX_OUT=outputs/bucket3_negerase_flux
export KIND_OVERRIDE_JSON='{"t3_text":{"edit_source":0.10,"edit_hedit":1.10},"t3_decal":{"edit_source":0.10,"edit_hedit":1.10}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

find "$PROJ/outputs/bucket3_neg_flux" "$PROJ/outputs/bucket3_negerase_flux" -path "*/seed_10/result.png" | wc -l
date
echo bucket3_ghost_complete
