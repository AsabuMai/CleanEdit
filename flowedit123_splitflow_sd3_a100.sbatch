#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH -J fe123-splitflow
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit123_splitflow_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPLITFLOW_DISABLE_LLM=1
hostname
date
cd "$PROJ/_baselines/src/SplitFlow"
"$PROJ/_baselines/envs/splitflow-py310/bin/python" run_script.py \
  --device_number 0 \
  --exp_yaml flowedit123_sd3_exp.yaml
date
