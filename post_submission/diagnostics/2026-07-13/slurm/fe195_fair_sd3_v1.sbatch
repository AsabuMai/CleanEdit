#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J fe195-fair
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe195_fair_sd3_v1_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"

export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MANIFEST=$PROJ/tmp/t1_sd3p195fix_manifest.json
export OUT=outputs/fe195_fair_sd3_v1
export SEED=${SEED:-10}
export MASK_OVERRIDE_JSON='{"parrots2":"/cluster/users/grad/2025/25t8103/project/tmp/masks/MASK_p195_tight2.png"}'
export KEY_OVERRIDE_JSON='{"parrots2":{"--prompt":"Two colorful red parrots perched on a rocky surface. Each parrot wears its own separate small ornate golden royal crown centered on top of its head, exactly two crowns total, one crown per bird. Preserve both birds, their faces, pose, rock, and background.","--edit-hedit-guidance-scale":"1.55","--rec-guidance-scale":"0.18","--struct-guidance-scale":"0.42","--tar-guidance-scale":"9.0","--object-mask-provider":"semantic"}}'

"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
