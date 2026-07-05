#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J b2r2-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket2_r2_sd3_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"

# Bucket 2 round 2: variant-B run put "CV PR" on the BLUE sign — glyph
# formation works, the gate was wrong (attention grounding covered both
# signs; the sign-only SAM masks were never attached in this manifest).
# Attach them via MASK_OVERRIDE_JSON, keep formation boost + weak preserve.
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket23_t3.json
export ABLATE=no_final_restore
export SEED=10
export SUBJECT_PRESERVE_MODE=1
export SUBJECT_OBJECT_PROVIDER=operation_support_v3
export SUBJECT_GROUNDING_METHOD=external_mask
export T3_REAL_EDIT=1
export T3_REAL_LOCAL_TARGET_SCALE=1.40
export T3_REAL_LOCAL_TARGET_CFG=9.0
export T3_REAL_REC_SCALE=0.12
export T3_REAL_STRUCT_SCALE=0.20
export T3_REAL_TRAJ_SCALE=0.06
export MASK_OVERRIDE_JSON='{"fe_238":"data/flowedit_compatible_118/sam_support_masks_all/fe_238_stop_arrow_1_cvpr_support.png","fe_119":"data/flowedit_compatible_118/sam_support_masks_all/fe_119_gas_station_2_iccv_support.png"}'
export OUT=outputs/bucket2_r2_sd3
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
find "$PROJ/$OUT" -path "*/seed_10/result.png" | wc -l
date
echo bucket2_r2_complete
