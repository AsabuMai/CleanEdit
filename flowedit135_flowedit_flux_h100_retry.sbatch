#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH -J fe135-flowedit-flux-r
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit135_flowedit_flux_retry_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
date
cd "$PROJ/_baselines/src/FlowEdit"
"$PROJ/.venv-h100/bin/python" run_script.py \
  --device_number 0 \
  --exp_yaml flowedit135_flux_exp.yaml
date
