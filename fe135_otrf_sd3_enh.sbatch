#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH -J fe135-otrf-enh
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_otrf_sd3_enh_%j.log
set -uo pipefail
export HF_HUB_CACHE=/cluster/users/grad/2025/25t8103/project/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=/cluster/users/grad/2025/25t8103/project/.cache/huggingface/hub
export HF_HOME=/cluster/users/grad/2025/25t8103/project/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
hostname; date
cd /cluster/users/grad/2025/25t8103/project/_baselines/src/OT-RF
/cluster/users/grad/2025/25t8103/project/_baselines/envs/ot-rf-otip-py310/bin/python run_script_enhanced.py \
  --device_number 0 --exp_yaml otrf135_sd3_enhanced_exp.yaml
date; echo fe135_otrf_sd3_enh_complete
