#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J b6r6-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket6_r6_flux_%j.log

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

# Bucket 6 round 6: meditation pair fe_172/175 only. r5 fixed the
# background (outside_lock 0.45) but the subject converts only below the
# waist: orange shirt + human head survive = inside-mask under-edit.
# Grid three escalations of inside-edit strength, background recipe fixed.
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket6_meditation.json
export ABLATE=no_final_restore
export SEED=10
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export T5_EXPAND_EDIT_MASK=0
export T5_POSE_PROMPT=1
export T5_AUTO_PROMPT_PREFIX=1
export T5_PROMPT_PREFIX=""
# A: pure fe_130 recipe transplant (hedit 1.55), r5 negative prompt unchanged
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark, duplicated figure, second person, oversized figure"
export FLUX_OUT=outputs/bucket6_r6a_flux
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45,"edit_hedit":1.55}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

# B: negative-prompt lever alone (bucket-3 lesson: suppress the SOURCE
# material through true-CFG), params identical to r5
export T5_NEGATIVE_PROMPT="photograph of a man, human skin, skin texture, orange t-shirt, fabric clothing, realistic human face, fur, text, watermark, duplicated figure, second person"
export FLUX_OUT=outputs/bucket6_r6b_flux
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

# C: combined + cut preserve budget + loosen struct hold
export FLUX_OUT=outputs/bucket6_r6c_flux
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45,"edit_hedit":1.55,"pbud":0.04,"struct":0.25}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

for d in bucket6_r6a_flux bucket6_r6b_flux bucket6_r6c_flux; do
  echo -n "$d: "; find "$PROJ/outputs/$d" -path "*/seed_10/result.png" | wc -l
done
date
echo bucket6_r6_complete
