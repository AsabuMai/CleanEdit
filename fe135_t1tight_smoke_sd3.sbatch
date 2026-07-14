#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J fe135-t1tight
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t1tight_smoke_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date; cd "$PROJ"
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t1tight_smoke2.json OUT=outputs/fe135_t1tight_smoke_sd3 SUBJECT_PRESERVE_MODE=1 SUBJECT_OBJECT_PROVIDER=operation_support_v3 SUBJECT_GROUNDING_METHOD=external_mask KIND_OVERRIDE_JSON='{"t1_accessory":{"struct":"0.65"}}' SUBJECT_KIND_OVERRIDE_JSON='{"t1_accessory":{"traj":"0.60","max_area":"0.18","dilate":"3"}}' "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
date; echo fe135_t1tight_smoke_sd3_complete
