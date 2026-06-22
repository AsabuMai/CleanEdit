#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J flux-pie
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_pie_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo "Refusing outside a100-01"; exit 2; }
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/pie_run.py"
date; echo flux_pie_complete
