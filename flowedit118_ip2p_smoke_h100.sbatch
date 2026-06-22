#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J fe118-ip2p-smoke
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit118_ip2p_smoke_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside h100-01" >&2
  exit 2
fi
"$PROJ/_baselines/envs-h100/instruct-pix2pix-py39/bin/python" run_flowedit118_traditional_baselines.py \
  --method instruct_pix2pix \
  --limit 1 \
  --steps 20 \
  --allow-download \
  --skip-complete
date
echo flowedit118_ip2p_smoke_complete
