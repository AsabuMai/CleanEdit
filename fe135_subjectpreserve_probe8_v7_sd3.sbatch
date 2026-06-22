#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J fe135-subjpres8-v7
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_subjectpreserve_probe8_v7_sd3_%j.log

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

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_subjectpreserve_probe8.json \
OUT=outputs/fe135_subjectpreserve_probe8_v7_sd3 \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
SUBJECT_TAR_GUIDANCE=8.0 \
SUBJECT_CORE_SCALE=0.90 \
SUBJECT_EDIT_SCALE=0.20 \
SUBJECT_TRAJ_SCALE=0.50 \
SUBJECT_SUPPORT_MAX_AREA=0.25 \
KIND_OVERRIDE_JSON='{"add":{"hedit":"0.85","text":"0.10"},"decal":{"hedit":"0.95","text":"0.14"},"material":{"hedit":"0.85","text":"0.14"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

date
echo fe135_subjectpreserve_probe8_v7_sd3_complete
