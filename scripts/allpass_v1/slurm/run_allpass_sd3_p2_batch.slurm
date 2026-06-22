#!/bin/bash
#SBATCH -J allpass_sd3_p2
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -t 24:00:00
#SBATCH -o logs/allpass_sd3_p2_%j.out
#SBATCH -e logs/allpass_sd3_p2_%j.err

set -euo pipefail
cd /cluster/users/grad/2025/25t8103/project
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp" || { hostname; exit 1; }
source .venv/bin/activate
python scripts/allpass_v1/run_allpass_batch.py \
  --backend sd3 \
  --gpu-only \
  --overwrite
