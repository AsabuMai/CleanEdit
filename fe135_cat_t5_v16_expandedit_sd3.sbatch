#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J fe135-cat-v16-expedit
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_cat_t5_v16_expandedit_sd3_%j.log

set -uo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
cd "$PROJ"

MASK=data/flowedit_compatible_135/derived_masks/fe_055_cat_7_wooden_sculpture_support_dilated_editmask.png

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_subjectpreserve_cat_t5probe1.json \
OUT=outputs/fe135_cat_t5_v16_expandedit_sd3 \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
SUBJECT_KIND_OVERRIDE_JSON='{"t5_material":{"source_v":"0.0","source_mask":"edit","source_steps":"12"}}' \
ADAPT_OVERRIDE_JSON='{"t5_material":{"--final-edit-mask":"'"$MASK"'","--final-edit-mask-mode":"replace"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

date
echo fe135_cat_t5_v16_expandedit_sd3_complete
