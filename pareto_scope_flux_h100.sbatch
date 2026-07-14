#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -J pareto-scope-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pareto_scope_flux_%j.log

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
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export SEED=10
export T5_EXPAND_EDIT_MASK=1
export T5_EXPAND_EDIT_RADIUS=21
export T5_EXPAND_EDIT_MASK_BLUR=6
export T5_POSE_PROMPT=1
export T5_AUTO_PROMPT_PREFIX=1
export T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark"
cd "$PROJ"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_pareto_non_t4_33.json

run_scope() {
  tag="$1"
  json="$2"
  export FLUX_OUT=outputs/pareto_sweep_20260701/scope_flux_${tag}
  export KIND_OVERRIDE_JSON="$json"
  "$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
}

run_scope g070 '{"t1_accessory":{"edit_hedit":0.665},"t2_insert":{"edit_hedit":0.665},"t3_text":{"edit_hedit":0.686},"t3_decal":{"edit_hedit":0.686},"t5_material":{"edit_hedit":0.91}}'
run_scope g085 '{"t1_accessory":{"edit_hedit":0.807},"t2_insert":{"edit_hedit":0.807},"t3_text":{"edit_hedit":0.833},"t3_decal":{"edit_hedit":0.833},"t5_material":{"edit_hedit":1.105}}'
run_scope g115 '{"t1_accessory":{"edit_hedit":1.092},"t2_insert":{"edit_hedit":1.092},"t3_text":{"edit_hedit":1.127},"t3_decal":{"edit_hedit":1.127},"t5_material":{"edit_hedit":1.495}}'
date
echo pareto_scope_flux_complete
