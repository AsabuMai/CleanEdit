#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH -J fe135-flux-t2
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_flux_t2_tune_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ"
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_flux_t2_tune7.json \
FLUX_OUT=outputs/fe135_flux_t2_tune/tokenfix_default \
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_kindonly.py

KIND_OVERRIDE_JSON='{"add":{"edit_hedit":1.18,"edit_region":0.44,"edit_target":0.28,"local_target":1.25,"rec":0.14,"struct":0.34,"traj":0.08,"region_transport":0.45,"pbud":0.28,"pgain":1.6,"core":84.0}}' \
ADAPT_OVERRIDE_JSON='{"add":{"--adaptive-edit-gain":"2.8","--tar-guidance-scale":"6.0","--edit-local-target-cfg-scale":"6.0"}}' \
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_flux_t2_tune7.json \
FLUX_OUT=outputs/fe135_flux_t2_tune/mild_add \
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_kindonly.py

KIND_OVERRIDE_JSON='{"add":{"edit_hedit":1.35,"edit_region":0.52,"edit_target":0.36,"local_target":1.45,"rec":0.10,"struct":0.30,"traj":0.05,"region_transport":0.55,"pbud":0.35,"pgain":1.2,"core":86.0}}' \
ADAPT_OVERRIDE_JSON='{"add":{"--adaptive-edit-gain":"3.5","--tar-guidance-scale":"6.5","--edit-local-target-cfg-scale":"6.5"}}' \
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_flux_t2_tune7.json \
FLUX_OUT=outputs/fe135_flux_t2_tune/strong_add \
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_kindonly.py
date; echo fe135_flux_t2_tune_complete
