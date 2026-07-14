#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH -J fe135-t3-gasabl
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_gas_ablate_v1_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub HF_HOME=$PROJ/.cache/huggingface HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date; cd "$PROJ"
env MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_gas_blank.json OUT=outputs/fe135_t3_gas_ablate_v1 START=0 LIMIT=4 SEED=10 SUBJECT_PRESERVE_MODE=1 SUBJECT_OBJECT_PROVIDER=operation_support_v3 SUBJECT_GROUNDING_METHOD=external_mask T3_REAL_EDIT=0 T3_REF_GUIDANCE=0 T3_SPELL_TEXT=0 T3_COMPACT_PROMPT=0 T3_SURFACE_PROMPT=0 ABLATE=preserve "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
echo "[gas_ablate count]"; find "$PROJ/outputs/fe135_t3_gas_ablate_v1" -name result.png | wc -l
date; echo gas_ablate_complete
