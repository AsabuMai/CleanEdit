#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J pie15-flux-bl
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie15_flux_bl_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DISABLE_BASELINE_NSFW=1
M=$PROJ/data/pie_pilot_20260618/manifest.json
OUTREL=outputs/pie15_baselines_20260618
hostname; date
echo "### FireFlow ###"
cd "$PROJ/_baselines/src/FireFlow/src"
MANIFEST=$M OUT=$OUTREL "$PROJ/_baselines/envs/fireflow-py310/bin/python" fireflow_batch.py || echo "fireflow errored"
date; echo "### RF-Solver ###"
cd "$PROJ/_baselines/src/RF-Solver-Edit/FLUX_Image_Edit/src"
MANIFEST=$M OUT=$OUTREL "$PROJ/_baselines/envs/rf-solver-edit-py310/bin/python" rfsolver_batch.py || echo "rfsolver errored"
date; echo pie15_flux_batch_complete
