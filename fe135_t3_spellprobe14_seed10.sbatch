#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --array=10
#SBATCH -J fe135-t3-spell
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_spellprobe14_seed%a_%A.log

set -uo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
SEED=${SLURM_ARRAY_TASK_ID}
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
echo "seed=$SEED"
cd "$PROJ"

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_spell_probe14.json \
OUT=outputs/fe135_t3_spellprobe_sd3_v1 \
SEED="$SEED" \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
T3_REAL_EDIT=1 \
T3_SPELL_TEXT=1 \
T3_SPELL_ATTENTION=1 \
T3_REAL_LOCAL_TARGET_SCALE=1.05 \
T3_REAL_LOCAL_TARGET_CFG=8.5 \
T3_REAL_REGION_TRANSPORT=0.36 \
T3_REAL_EDIT_TARGET_SCALE=0.24 \
T3_REAL_EDIT_REGION_SCALE=0.38 \
T3_REAL_REC_SCALE=0.20 \
T3_REAL_STRUCT_SCALE=0.28 \
T3_REAL_TRAJ_SCALE=0.08 \
T3_REAL_RELATION=on_surface \
T3_REAL_LAYERING=object_contact \
SUBJECT_KIND_OVERRIDE_JSON='{"t3_text":{"tar":"8.0","core":"1.05","subj":"0.20","traj":"0.0","max_area":"0.42","min_area":"0.015","relation":"on_surface","layering":"object_contact","keep":"3","dilate":"5"},"t3_decal":{"tar":"8.0","core":"0.95","subj":"0.16","traj":"0.0","max_area":"0.35","min_area":"0.015","relation":"on_surface","layering":"object_contact","keep":"3","dilate":"5"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

echo "[sd3 count seed_$SEED]"
find "$PROJ/outputs/fe135_t3_spellprobe_sd3_v1" -path "*/seed_${SEED}/result.png" | wc -l
date

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_spell_probe14.json \
FLUX_OUT=outputs/fe135_t3_spellprobe_flux_v1 \
SEED="$SEED" \
FLUX_STEPS=12 \
FLUX_N_MAX=10 \
T3_SPELL_TEXT=1 \
KIND_OVERRIDE_JSON='{"decal":{"edit_hedit":1.05,"edit_region":0.42,"edit_target":0.28,"local_target":1.10,"region_transport":0.35,"rec":0.22,"struct":0.36,"traj":0.12,"pbud":0.12,"pgain":3.0}}' \
"$PROJ/.venv/bin/python" scripts/pie_batch_flux_kindaware_v11.py

echo "[flux count seed_$SEED]"
find "$PROJ/outputs/fe135_t3_spellprobe_flux_v1" -path "*/seed_${SEED}/result.png" | wc -l
date
echo "fe135_t3_spellprobe14_seed${SEED}_complete"
