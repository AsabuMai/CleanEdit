#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --array=10
#SBATCH -J fe135-t3-cflux2
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_compactflux2_seed%a_%A.log

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

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_oldtext_probe14.json \
FLUX_OUT=outputs/fe135_t3_compactflux_probe_v1 \
START=0 \
LIMIT=2 \
SEED="$SEED" \
FLUX_STEPS=12 \
FLUX_N_MAX=10 \
FLUX_MAX_SEQ=96 \
FLUX_OFFLOAD=sequential \
FLUX_KEEP_TEXT_ENCODERS_CPU=1 \
FLUX_PROMPT_ENCODE_DEVICE=cpu \
T3_COMPACT_PROMPT=1 \
T3_SURFACE_PROMPT=0 \
T3_SPELL_TEXT=0 \
KIND_OVERRIDE_JSON='{"decal":{"edit_hedit":0.98,"edit_region":0.34,"edit_target":0.22,"local_target":0.98,"region_transport":0.24,"rec":0.26,"struct":0.42,"traj":0.18,"pbud":0.12,"pgain":3.0,"outside_lock":0.02}}' \
"$PROJ/.venv/bin/python" scripts/pie_batch_flux_kindaware_v11.py

echo "[compact flux count seed_$SEED]"
find "$PROJ/outputs/fe135_t3_compactflux_probe_v1" -path "*/seed_${SEED}/result.png" | wc -l
date
echo "fe135_t3_compactflux2_seed${SEED}_complete"
