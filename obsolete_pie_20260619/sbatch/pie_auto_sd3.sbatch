#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J pie-sd3-auto
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie_auto_sd3_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
hostname; date
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/pie_auto_sd3.py"
date; echo pie_auto_sd3_complete
