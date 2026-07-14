#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=05:30:00
#SBATCH --array=11-12%1
#SBATCH -J fe135-sd3-ms
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_subjectpreserve_full_v10_sd3_seed%a_%A.log

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

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_sam_135.json \
OUT=outputs/fe135_subjectpreserve_full_v10_sd3 \
SEED="$SEED" \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
KIND_OVERRIDE_JSON='{"t5_material":{"hedit":"0.82","text":"0.14","rec":"0.62","struct":"0.75"}}' \
SUBJECT_KIND_OVERRIDE_JSON='{"t5_material":{"tar":"7.0","core":"0.85","subj":"0.40","traj":"0.70","max_area":"0.45","min_area":"0.10","relation":"inside_host","layering":"none","keep":"2","dilate":"5"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

echo "[count seed_$SEED]"
find "$PROJ/outputs/fe135_subjectpreserve_full_v10_sd3" -path "*/seed_${SEED}/result.png" | wc -l
date
echo "fe135_subjectpreserve_full_v10_sd3_seed${SEED}_complete"
