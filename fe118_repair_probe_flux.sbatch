#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH -J fe118-repair-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe118_repair_flux_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ"
KIND_OVERRIDE_JSON="$(cat $PROJ/repair_flux_kind.json)" \
ADAPT_OVERRIDE_JSON="$(cat $PROJ/repair_flux_adapt.json)" \
MANIFEST=$PROJ/data/flowedit_compatible_118/manifest_repair_probe.json \
FLUX_OUT=outputs/fe118_repair_probe_dece_flux \
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_repair.py
date; echo fe118_repair_probe_flux_complete
