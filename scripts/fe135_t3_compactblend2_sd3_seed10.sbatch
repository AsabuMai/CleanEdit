#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --array=0-1%1
#SBATCH -J fe135-t3-cblend2-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_compactblend2_sd3_idx%a_%A.log

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

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_oldtext_probe14.json \
OUT=outputs/fe135_t3_compactblend_sd3_probe_v1 \
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
T3_REAL_LOCAL_TARGET_SCALE=1.10 \
T3_REAL_LOCAL_TARGET_CFG=8.5 \
T3_REAL_REGION_TRANSPORT=0.34 \
T3_REAL_EDIT_TARGET_SCALE=0.26 \
T3_REAL_EDIT_REGION_SCALE=0.40 \
T3_REAL_REC_SCALE=0.18 \
T3_REAL_STRUCT_SCALE=0.30 \
T3_REAL_TRAJ_SCALE=0.08 \
T3_REAL_RELATION=on_surface \
T3_REAL_LAYERING=object_contact \
SUBJECT_KIND_OVERRIDE_JSON='{"t3_text":{"tar":"8.5","core":"1.10","subj":"0.18","traj":"0.0","max_area":"0.20","min_area":"0.002","relation":"on_surface","layering":"object_contact","keep":"4","dilate":"3"},"t3_decal":{"tar":"8.5","core":"1.00","subj":"0.14","traj":"0.0","max_area":"0.20","min_area":"0.002","relation":"on_surface","layering":"object_contact","keep":"4","dilate":"3"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

echo "[sd3 compact blend count seed_$SEED]"
find "$PROJ/outputs/fe135_t3_compactblend_sd3_probe_v1" -path "*/seed_${SEED}/result.png" | wc -l
date
echo "fe135_t3_compactblend2_sd3_idx${IDX}_complete"
