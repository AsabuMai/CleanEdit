#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH -J pie-bl-dev
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie_bl_dev_%j.log
set -uo pipefail
cd /cluster/users/grad/2025/25t8103/project; mkdir -p logs
export HF_HUB_CACHE=/cluster/users/grad/2025/25t8103/project/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MANIFEST=/cluster/users/grad/2025/25t8103/project/data/pie_dev_20260618/manifest.json
hostname; date
for M in fireflow rf_solver_edit reflex; do
  echo "######## DEV $M ########"; date
  METHOD=$M OUT=outputs/pie_dev_baselines_20260618 /cluster/users/grad/2025/25t8103/project/.venv/bin/python scripts/pie_baselines.py || echo "$M driver errored"
done
date; echo pie_bl_dev_complete
