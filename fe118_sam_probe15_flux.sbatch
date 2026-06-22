#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH -J fe118-sam-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe118_sam_probe15_flux_%j.log

set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
cd "$PROJ" || exit 1
KIND_OVERRIDE_JSON="$(cat $PROJ/r3_flux_kind.json)" \
ADAPT_OVERRIDE_JSON="$(cat $PROJ/r3_flux_adapt.json)" \
MANIFEST=$PROJ/data/flowedit_compatible_118/manifest_sam_probe15.json \
FLUX_OUT=outputs/fe118_sam_probe15_dece_flux \
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_kindonly.py
date
echo fe118_sam_probe15_flux_complete
