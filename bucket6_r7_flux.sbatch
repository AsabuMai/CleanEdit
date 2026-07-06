#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J b6r7-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket6_r7_flux_%j.log

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

# Bucket 6 round 7: canonical fully converts the subject (expansion ON,
# plain prompt) but wrecks the background; our r5/r6 recipe protects the
# background but strangles the conversion (statue head cannot fit inside
# the tight human silhouette; r6 chimera concentrates at mask boundary).
# Hypothesis: canonical recipe + outside_lock 0.45 alone = both goals.
# All variants: plain prompt (pose/auto-prefix OFF), expansion ON (default).
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket6_meditation.json
export ABLATE=no_final_restore
export SEED=10
export FLUX_LOCAL_FILES_ONLY=1
export FLUX_OFFLOAD=model
export FLUX_STEPS=12
export FLUX_N_MAX=10
export T5_POSE_PROMPT=0
export T5_AUTO_PROMPT_PREFIX=0
export T5_PROMPT_PREFIX=""

# D: canonical + outside_lock 0.45 only (minimal delta from canonical)
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark, duplicated figure, second person, oversized figure"
export FLUX_OUT=outputs/bucket6_r7d_flux
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

# E: D + anti-human negative prompt (r6's proven conversion lever)
export T5_NEGATIVE_PROMPT="photograph of a man, human skin, skin texture, orange t-shirt, fabric clothing, realistic human face, fur, text, watermark, duplicated figure, second person"
export FLUX_OUT=outputs/bucket6_r7e_flux
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

# F: E + hedit 1.55
export FLUX_OUT=outputs/bucket6_r7f_flux
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45,"edit_hedit":1.55}}'
"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py

for d in bucket6_r7d_flux bucket6_r7e_flux bucket6_r7f_flux; do
  echo -n "$d: "; find "$PROJ/outputs/$d" -path "*/seed_10/result.png" | wc -l
done
date
echo bucket6_r7_complete
