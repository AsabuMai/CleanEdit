#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH -J fe118-otrf-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe118_otrf_sd3_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
hostname; date
cd "$PROJ/_baselines/src/OT-RF"
"$PROJ/_baselines/envs/ot-rf-otip-py310/bin/python" run_script.py --device_number 0 --exp_yaml otrf118_sd3_exp.yaml
date; echo fe118_otrf_sd3_complete
