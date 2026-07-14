#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J b2-blank
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket2_blank_sd3_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"

# Bucket 2 (SD3 text BLANK): erase succeeds, glyph formation fails
# (preserve-first collapse). Try boosting local-target formation, then
# formation boost + weakened preserve inside the run.
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket23_t3.json
export ABLATE=no_final_restore
export SEED=10
export SUBJECT_PRESERVE_MODE=1
export SUBJECT_OBJECT_PROVIDER=operation_support_v3
export SUBJECT_GROUNDING_METHOD=external_mask
export T3_REAL_EDIT=1

# Variant A: stronger local-target formation.
export T3_REAL_LOCAL_TARGET_SCALE=1.40
export T3_REAL_LOCAL_TARGET_CFG=9.0
export OUT=outputs/bucket2_lt_sd3
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

# Variant B: formation boost + weaker preserve (rec/struct/traj down).
export T3_REAL_REC_SCALE=0.12
export T3_REAL_STRUCT_SCALE=0.20
export T3_REAL_TRAJ_SCALE=0.06
export OUT=outputs/bucket2_ltweak_sd3
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

find "$PROJ/outputs/bucket2_lt_sd3" "$PROJ/outputs/bucket2_ltweak_sd3" -path "*/seed_10/result.png" | wc -l
date
echo bucket2_blank_complete
