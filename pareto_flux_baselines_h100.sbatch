#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH -J pareto-flux-base
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pareto_flux_baselines_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"

cd "$PROJ/_baselines/src/FlowEdit"
"$PROJ/.venv-h100/bin/python" run_script.py --device_number 0 --exp_yaml pareto_flux_exp.yaml

cd "$PROJ"
for spec in g070:2.45 g085:2.975 g100:3.5 g115:4.025; do
  tag=${spec%%:*}; val=${spec##*:}
  MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_pareto_non_t4_33.json \
  OUT=outputs/pareto_sweep_20260701/reflex_${tag} METHOD=reflex SEED=10 REFLEX_GUIDANCE=$val \
  "$PROJ/_baselines/envs/reflex-py310/bin/python" scripts/flowedit_reflex_sweep_batch.py
done

for spec in g070:1.4 g085:1.7 g100:2.0 g115:2.3; do
  tag=${spec%%:*}; val=${spec##*:}
  MANIFEST_CSV=data/flowedit_compatible_135/baseline_pareto_fireflow.csv \
  OUT=outputs/pareto_sweep_20260701/fireflow_${tag} METHOD=fireflow SEED=10 FIREFLOW_GUIDANCE=$val \
  "$PROJ/_baselines/envs/fireflow-py310/bin/python" scripts/run_fireflow_sweep_batch.py
done
date
echo pareto_flux_baselines_complete
