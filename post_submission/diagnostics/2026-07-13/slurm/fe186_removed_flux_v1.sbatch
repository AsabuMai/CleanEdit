#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:nvidia_h100_pcie:6
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH -J fe186-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe186_removed_flux_v1_%j.log

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
export MANIFEST=$PROJ/tmp/fe186_removed_mask_v1_manifest.json
export FLUX_OUT=outputs/fe186_removed_flux_v3
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export SEED=10
export KEY_OVERRIDE_JSON='{"muffins_1":{"--prompt":"A plate of the same five white muffins. Replace every existing raspberry in place with one whole fresh red strawberry with a green leafy cap. Preserve the exact muffins, plate, stand, layout, camera, and background; change only the fruit.","--negative-prompt":"raspberry, raspberries, cupcake liner, paper cup, extra muffin, sliced fruit, gray blob, green blob, distorted muffin"}}'

"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
