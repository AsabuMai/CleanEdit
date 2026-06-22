#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J pie-bl-h100test
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie_bl_h100test_%j.log
set -uo pipefail
cd /cluster/users/grad/2025/25t8103/project; mkdir -p logs
export HF_HUB_CACHE=/cluster/users/grad/2025/25t8103/project/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MANIFEST=/cluster/users/grad/2025/25t8103/project/data/pie_dev_20260618/manifest.json
hostname; date
"$PWD/_baselines/envs/fireflow-py310/bin/python" -c "import torch; print('torch',torch.__version__,'cuda_ok',torch.cuda.is_available(),'dev',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
METHOD=fireflow LIMIT=1 OUT=outputs/pie_h100test_20260618 /cluster/users/grad/2025/25t8103/project/.venv/bin/python scripts/pie_baselines.py || echo "fireflow h100 errored"
date; echo h100test_complete
