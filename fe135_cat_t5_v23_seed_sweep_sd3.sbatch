#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:10:00
#SBATCH -J fe135-cat-v23-seeds
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_cat_t5_v23_seed_sweep_sd3_%j.log

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

for S in 11 12; do
  MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_subjectpreserve_cat_t5probe1.json \
  OUT=outputs/fe135_cat_t5_v23_seed_sweep_sd3 \
  SEED=$S \
  SUBJECT_PRESERVE_MODE=1 \
  SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
  SUBJECT_GROUNDING_METHOD=external_mask \
  T5_PROMPT_PREFIX="A smooth clean-edged wooden cat sculpture with no fur, no whiskers, and no floating hair strands" \
  T5_EXPAND_EDIT_RADIUS=21 \
  T5_EXPAND_EDIT_MASK_BLUR=6 \
  "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
done

date
echo fe135_cat_t5_v23_seed_sweep_sd3_complete
