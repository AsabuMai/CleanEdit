#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J fe118-sam-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe118_sam_probe15_sd3_%j.log

set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
cd "$PROJ" || exit 1
KIND_OVERRIDE_JSON="$(cat $PROJ/r3_sd3_kind.json)" \
ADAPT_OVERRIDE_JSON="$(cat $PROJ/r3_sd3_adapt.json)" \
MANIFEST=$PROJ/data/flowedit_compatible_118/manifest_sam_probe15.json \
OUT=outputs/fe118_sam_probe15_dece_sd3 \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware_kindonly.py
date
echo fe118_sam_probe15_sd3_complete
