#!/bin/bash
#SBATCH -J allpass_flux_p2
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -t 24:00:00
#SBATCH -o logs/allpass_flux_p2_%j.out
#SBATCH -e logs/allpass_flux_p2_%j.err

set -euo pipefail
cd /cluster/users/grad/2025/25t8103/project
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp" || { hostname; exit 1; }
source .venv-h100/bin/activate
python scripts/allpass_v1/run_allpass_batch.py \
  --backend flux \
  --gpu-only \
  --overwrite
