#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --array=0-1%1
#SBATCH -J fe135-t3-panelglyph
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_panelglyph_gas2_sd3_idx%a_%A.log

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

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_panelglyph_gas2_v1.json \
OUT=outputs/fe135_t3_panelglyph_sd3_probe_v1 \
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
T3_REF_GUIDANCE_SCALE=0.20 \
T3_REF_CHROMA_MODE=yuv_direction \
T3_REF_CHROMA_SCALE=0.10 \
T3_REF_LUMA_SCALE=0.08 \
T3_REF_GRAD_SCALE=0.12 \
T3_REF_MAX_STRUCT_RATIO=0.10 \
T3_REF_PROJECT_CONFLICT=0.60 \
T3_REF_DARKNESS_GUARD=0.70 \
T3_REF_DARKNESS_MARGIN=0.02 \
T3_REF_SMOOTH_KERNEL=3 \
T3_REF_LOWFREQ_KERNEL=17 \
T3_REF_LOWFREQ_SCALE=0.55 \
T3_REF_SCHEDULE_START=0.15 \
T3_REF_SCHEDULE_STOP=0.72 \
T3_REF_SCHEDULE_POWER=1.0 \
T3_REAL_LOCAL_TARGET_SCALE=1.05 \
T3_REAL_LOCAL_TARGET_CFG=8.5 \
T3_REAL_REGION_TRANSPORT=0.36 \
T3_REAL_EDIT_TARGET_SCALE=0.28 \
T3_REAL_EDIT_REGION_SCALE=0.42 \
T3_REAL_REC_SCALE=0.20 \
T3_REAL_STRUCT_SCALE=0.32 \
T3_REAL_TRAJ_SCALE=0.06 \
T3_REAL_RELATION=on_surface \
T3_REAL_LAYERING=object_contact \
SUBJECT_KIND_OVERRIDE_JSON='{"t3_text":{"tar":"8.0","core":"1.00","subj":"0.12","traj":"0.0","max_area":"0.24","min_area":"0.002","relation":"on_surface","layering":"object_contact","keep":"4","dilate":"2"},"t3_decal":{"tar":"8.0","core":"0.92","subj":"0.10","traj":"0.0","max_area":"0.24","min_area":"0.002","relation":"on_surface","layering":"object_contact","keep":"4","dilate":"2"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

echo "[sd3 panelglyph count seed_$SEED]"
find "$PROJ/outputs/fe135_t3_panelglyph_sd3_probe_v1" -path "*/seed_${SEED}/result.png" | wc -l
date
echo "fe135_t3_panelglyph_gas2_sd3_idx${IDX}_complete"
