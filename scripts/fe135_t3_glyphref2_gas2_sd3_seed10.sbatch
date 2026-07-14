#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --array=0-1%1
#SBATCH -J fe135-t3-gref2b-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_glyphref2_gas2_sd3_idx%a_%A.log

set -uo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
IDX=${SLURM_ARRAY_TASK_ID}
SEED=10
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
echo "idx=$IDX seed=$SEED"
cd "$PROJ"

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_glyphref_gas2_v2.json \
OUT=outputs/fe135_t3_glyphref_sd3_probe_v2 \
START="$IDX" \
LIMIT=1 \
SEED="$SEED" \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
T3_REAL_EDIT=1 \
T3_COMPACT_PROMPT=1 \
T3_SURFACE_PROMPT=0 \
T3_SPELL_TEXT=1 \
T3_REF_GUIDANCE=1 \
T3_REF_GUIDANCE_SCALE=0.32 \
T3_REF_CHROMA_SCALE=0.55 \
T3_REF_LUMA_SCALE=0.04 \
T3_REF_GRAD_SCALE=0.04 \
T3_REF_MAX_STRUCT_RATIO=0.16 \
T3_REAL_LOCAL_TARGET_SCALE=0.90 \
T3_REAL_LOCAL_TARGET_CFG=8.5 \
T3_REAL_REGION_TRANSPORT=0.30 \
T3_REAL_EDIT_TARGET_SCALE=0.20 \
T3_REAL_EDIT_REGION_SCALE=0.32 \
T3_REAL_REC_SCALE=0.24 \
T3_REAL_STRUCT_SCALE=0.38 \
T3_REAL_TRAJ_SCALE=0.10 \
T3_REAL_RELATION=on_surface \
T3_REAL_LAYERING=object_contact \
SUBJECT_KIND_OVERRIDE_JSON='{"t3_text":{"tar":"8.0","core":"1.02","subj":"0.16","traj":"0.0","max_area":"0.20","min_area":"0.002","relation":"on_surface","layering":"object_contact","keep":"4","dilate":"3"},"t3_decal":{"tar":"8.0","core":"0.96","subj":"0.12","traj":"0.0","max_area":"0.20","min_area":"0.002","relation":"on_surface","layering":"object_contact","keep":"4","dilate":"3"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

echo "[sd3 glyphref2 count seed_$SEED]"
find "$PROJ/outputs/fe135_t3_glyphref_sd3_probe_v2" -path "*/seed_${SEED}/result.png" | wc -l
date
echo "fe135_t3_glyphref2_gas2_sd3_idx${IDX}_complete"
