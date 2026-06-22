#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -J pie-eval
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie_eval_%j.log
set -euo pipefail
cd /cluster/users/grad/2025/25t8103/project; mkdir -p logs
export HF_HOME=/cluster/users/grad/2025/25t8103/project/.cache/huggingface
export TRANSFORMERS_CACHE=/cluster/users/grad/2025/25t8103/project/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
hostname; date
.venv/bin/python scripts/pie_eval.py
date; echo pie_eval_complete
