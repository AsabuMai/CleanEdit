#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=09:00:00
#SBATCH -J fe135-ledits
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit135_ledits_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ"
if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside h100-01" >&2
  exit 2
fi
"$PROJ/_baselines/envs-h100/ledits-pp-py39/bin/python" run_flowedit118_traditional_baselines.py \
  --manifest "$PROJ/data/flowedit_compatible_135/manifest.json" \
  --output-dir "$PROJ/outputs/flowedit135_traditional_baselines" \
  --method ledits_pp \
  --dtype float32 \
  --inversion-steps 30 \
  --skip-complete \
  --allow-download
date; echo flowedit135_ledits_complete
