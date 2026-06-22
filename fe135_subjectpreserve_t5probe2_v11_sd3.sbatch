#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH -J fe135-t5pose-v11
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_subjectpreserve_t5probe2_v11_sd3_%j.log

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
OUT=outputs/fe135_subjectpreserve_t5probe2_v11_sd3 \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
KIND_OVERRIDE_JSON='{"t5_material":{"hedit":"0.86","text":"0.14","rec":"0.55","struct":"0.70"}}' \
SUBJECT_KIND_OVERRIDE_JSON='{"t5_material":{"tar":"7.0","core":"0.90","subj":"0.50","traj":"0.55","max_area":"0.45","min_area":"0.10","relation":"inside_host","layering":"none","keep":"2","dilate":"5"}}' \
ADAPT_OVERRIDE_JSON='{"t5_material":{"--source-inject-v-scale":"0.25","--source-inject-mask-mode":"edit","--source-inject-steps":"12"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

date
echo fe135_subjectpreserve_t5probe2_v11_sd3_complete
