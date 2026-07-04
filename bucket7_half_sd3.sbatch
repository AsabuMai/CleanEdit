#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J b7-half-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket7_half_sd3_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"

# Bucket 7 (HALF): DINO detected only the corgi head as "dog" (box y=48-236
# of 512). Keep all boxes at a lower threshold so a full-body proposal joins
# the union; if the head-only failure persists this becomes a mask-fills-box
# fallback task.
KEEP_ALL_BOXES=1 BOX_THRESHOLD=0.13 BOX_SCORE_FLOOR=0.14 BOX_SCORE_REL=0.35 \
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket7_src.json \
OUT_DIR=$PROJ/data/flowedit_compatible_118/sam_support_masks_plural_v2 \
OUT_MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket7_half.json \
"$PROJ/.venv/bin/python" scripts/make_flowedit118_sam_support_manifest.py

export ABLATE=no_final_restore
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket7_half.json
export OUT=outputs/bucket7_half_sd3
export SEED=10
export SUBJECT_PRESERVE_MODE=1
export SUBJECT_OBJECT_PROVIDER=operation_support_v3
export SUBJECT_GROUNDING_METHOD=external_mask
export KIND_OVERRIDE_JSON='{"t5_material":{"hedit":"0.82","text":"0.14","rec":"0.62","struct":"0.75"}}'
export SUBJECT_KIND_OVERRIDE_JSON='{"t5_material":{"tar":"7.0","core":"0.85","subj":"0.40","traj":"0.70","max_area":"0.45","min_area":"0.10","relation":"inside_host","layering":"none","keep":"2","dilate":"5"}}'
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
find "$PROJ/$OUT" -path "*/seed_10/result.png" | wc -l
date
echo bucket7_half_sd3_complete
