#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J probe-t5t1
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/probe_t5t1_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
cd "$PROJ"
MAN=$PROJ/data/flowedit_compatible_118/manifest_probe_t5t1.json
ARM="${ARM:?set ARM=A or B}"
ADAPT=$PROJ/probe_t5t1_adapt${ARM}.json
echo "=== probe arm $ARM, adapt=$ADAPT ==="
KIND_OVERRIDE_JSON="$(cat $PROJ/r3_sd3_kind.json)" \
ADAPT_OVERRIDE_JSON="$(cat $ADAPT)" \
MANIFEST=$MAN \
OUT=outputs/probe_t5t1_sd3_${ARM} \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
date; echo probe_t5t1_sd3_${ARM}_complete
