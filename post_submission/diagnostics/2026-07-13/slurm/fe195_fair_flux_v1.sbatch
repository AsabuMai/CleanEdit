#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:nvidia_h100_pcie:6
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH -J fe195-fair-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe195_fair_flux_v1_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
cd "$PROJ"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MANIFEST=$PROJ/tmp/t1_sd3p195fix_manifest.json
export FLUX_OUT=outputs/fe195_fair_flux_v1
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export SEED=10
export MASK_OVERRIDE_JSON='{"parrots2":"/cluster/users/grad/2025/25t8103/project/tmp/masks/MASK_p195_crowns_up_v2.png"}'
export KEY_OVERRIDE_JSON='{"parrots2":{"--prompt":"Two colorful red parrots perched on a rocky surface. Each parrot wears its own separate small ornate golden royal crown centered on top of its head, exactly two crowns total, one crown per bird. Preserve both birds, their faces, pose, rock, and background."}}'

"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
