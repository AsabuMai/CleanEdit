#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J pie-flux-batch
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie_batch_flux_%j.log
set -euo pipefail
cd /cluster/users/grad/2025/25t8103/project; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
hostname; date
.venv-h100/bin/python scripts/pie_batch_flux.py ${LIMIT:-2}
date; echo pie_batch_flux_complete
