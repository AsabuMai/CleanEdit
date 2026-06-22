#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=11:00:00
#SBATCH -J fe135-flowedit-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit135_flowedit_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date
cd "$PROJ/_baselines/src/FlowEdit"
"$PROJ/_baselines/envs/flowedit-py310/bin/python3" run_script.py \
  --device_number 0 \
  --exp_yaml flowedit135_sd3_exp.yaml
date; echo flowedit135_flowedit_sd3_complete
