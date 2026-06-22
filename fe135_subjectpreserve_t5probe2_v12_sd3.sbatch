#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH -J fe135-t5pose-v12
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_subjectpreserve_t5probe2_v12_sd3_%j.log

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

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_subjectpreserve_t5probe2_poseprompt.json \
OUT=outputs/fe135_subjectpreserve_t5probe2_v12_sd3 \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
T5_REF_ROOT=outputs/fe135_subjectpreserve_probe8_v9_sd3 \
T5_REF_SCALE=0.20 \
T5_REF_CHROMA_MODE=yuv_direction \
T5_REF_CHROMA_MAG=0.55 \
T5_REF_LUMA=0.85 \
T5_REF_GRAD=0.45 \
T5_REF_MAX_STRUCT_RATIO=0.45 \
KIND_OVERRIDE_JSON='{"t5_material":{"hedit":"0.82","text":"0.12","rec":"0.58","struct":"0.72"}}' \
SUBJECT_KIND_OVERRIDE_JSON='{"t5_material":{"tar":"7.0","core":"0.85","subj":"0.40","traj":"0.65","max_area":"0.45","min_area":"0.10","relation":"inside_host","layering":"none","keep":"2","dilate":"5"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

date
echo fe135_subjectpreserve_t5probe2_v12_sd3_complete
