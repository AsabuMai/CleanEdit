#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J fe186-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe186_removed_sd3_v1_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MANIFEST=$PROJ/tmp/fe186_removed_mask_v1_manifest.json
export OUT=outputs/fe186_removed_sd3_v3
export SEED=10
export KEY_OVERRIDE_JSON='{"muffins_1":{"--prompt":"A plate of the same five white muffins. Replace every existing raspberry in place with one whole fresh red strawberry with a green leafy cap. Preserve the exact muffins, plate, stand, layout, camera, and background; change only the fruit.","--negative-prompt":"raspberry, raspberries, cupcake liner, paper cup, extra muffin, sliced fruit, gray blob, green blob, distorted muffin"}}'

"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
