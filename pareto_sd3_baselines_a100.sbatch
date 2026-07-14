#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH -J pareto-sd3-base
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pareto_sd3_baselines_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export SPLITFLOW_DISABLE_LLM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"

cd "$PROJ/_baselines/src/FlowEdit"
"$PROJ/_baselines/envs/flowedit-py310/bin/python3" run_script.py --device_number 0 --exp_yaml pareto_sd3_exp.yaml

cd "$PROJ/_baselines/src/SplitFlow"
"$PROJ/_baselines/envs/splitflow-py310/bin/python" run_script.py --device_number 0 --exp_yaml pareto_sd3_exp.yaml

cd "$PROJ/_baselines/src/OT-RF"
"$PROJ/_baselines/envs/ot-rf-otip-py310/bin/python" run_script_enhanced.py --device_number 0 --exp_yaml pareto_sd3_enhanced_exp.yaml
date
echo pareto_sd3_baselines_complete
