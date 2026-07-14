#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH -J pareto-scope-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pareto_scope_sd3_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_pareto_non_t4_33.json
export SUBJECT_PRESERVE_MODE=1
export SUBJECT_OBJECT_PROVIDER=operation_support_v3
export SUBJECT_GROUNDING_METHOD=external_mask
export SUBJECT_KIND_OVERRIDE_JSON='{"t5_material":{"tar":"7.0","core":"0.85","subj":"0.40","traj":"0.70","max_area":"0.45","min_area":"0.10","relation":"inside_host","layering":"none","keep":"2","dilate":"5"}}'

run_scope() {
  tag="$1"
  json="$2"
  OUT=outputs/pareto_sweep_20260701/scope_sd3_${tag} \
  KIND_OVERRIDE_JSON="$json" \
  "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
}

run_scope g070 '{"t1_accessory":{"hedit":"0.77"},"t2_insert":{"hedit":"0.665"},"t3_text":{"hedit":"0.735"},"t3_decal":{"hedit":"0.595"},"t5_material":{"hedit":"0.574","text":"0.14","rec":"0.62","struct":"0.75"}}'
run_scope g085 '{"t1_accessory":{"hedit":"0.935"},"t2_insert":{"hedit":"0.807"},"t3_text":{"hedit":"0.892"},"t3_decal":{"hedit":"0.722"},"t5_material":{"hedit":"0.697","text":"0.14","rec":"0.62","struct":"0.75"}}'
run_scope g115 '{"t1_accessory":{"hedit":"1.265"},"t2_insert":{"hedit":"1.092"},"t3_text":{"hedit":"1.208"},"t3_decal":{"hedit":"0.977"},"t5_material":{"hedit":"0.943","text":"0.14","rec":"0.62","struct":"0.75"}}'
date
echo pareto_scope_sd3_complete
