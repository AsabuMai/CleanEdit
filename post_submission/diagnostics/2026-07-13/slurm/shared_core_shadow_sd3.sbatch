#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH -J sd3-core-shadow
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/shared_core_shadow_sd3_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"

export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket23_t3.json
export START=0
export LIMIT=1
export OUT=${OUT:-outputs/shared_core_shadow_sd3_probe}
export SEED=10
export SD3_SHARED_CORE_SHADOW=${SD3_SHARED_CORE_SHADOW:-1}
export KIND_OVERRIDE_JSON='{"t3_text":{"text":"0"},"t3_decal":{"text":"0"}}'

"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
test -f "$PROJ/$OUT/fe_119_gas_station_2_iccv/support_v3_controller_rmsgap/seed_10/stats.json"
echo shared_core_shadow_complete
