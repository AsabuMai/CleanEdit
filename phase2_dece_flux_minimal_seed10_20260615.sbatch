#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH -J p2-flux-min
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_dece_flux_minimal_%j.log

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project

export HF_HOME="$PROJECT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PROJECT/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

host="$(hostname)"
echo "host=$host"
date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

.venv/bin/python scripts/run_phase2_dece_flux_minimal.py \
  --skip-existing

date
echo phase2_dece_flux_minimal_seed10_complete
