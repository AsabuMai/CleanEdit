#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J b6r8-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket6_r8_flux_%j.log

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

# Bucket 6 round 8 (falsification round): r7 showed preserve-vs-transform
# competition resolves by spatial separation (chimera halves / side-by-side
# doubles). Zero out every preserve term INSIDE a tight mask, keep the
# background on outside_lock alone. If the orange shirt still survives,
# the attachment lives in the inversion trajectory itself, not guidance.
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket6_meditation.json
export ABLATE=no_final_restore
export SEED=10
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export T5_EXPAND_EDIT_MASK=0
export T5_POSE_PROMPT=0
export T5_AUTO_PROMPT_PREFIX=0
export T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="photograph of a man, human skin, skin texture, orange t-shirt, fabric clothing, realistic human face, fur, text, watermark, duplicated figure, second person"

# G: zero preserve inside (struct too)
export FLUX_OUT=outputs/bucket6_r8g_flux
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45,"edit_hedit":1.55,"traj":0.0,"pbud":0.0,"struct":0.0,"edit_anchor":0.02,"edit_source":0.0,"rec":0.02}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

# H: same but keep struct 0.40 (hold pose via structure only)
export FLUX_OUT=outputs/bucket6_r8h_flux
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45,"edit_hedit":1.55,"traj":0.0,"pbud":0.0,"struct":0.40,"edit_anchor":0.02,"edit_source":0.0,"rec":0.02}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

for d in bucket6_r8g_flux bucket6_r8h_flux; do
  echo -n "$d: "; find "$PROJ/outputs/$d" -path "*/seed_10/result.png" | wc -l
done
date
echo bucket6_r8_complete
