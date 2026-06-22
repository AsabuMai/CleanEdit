#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:20:00
#SBATCH -J fe135-cat-flux-v2
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_cat_t5_flux_v2_seq_h100_%j.log

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

hostname
date
cd "$PROJ"

if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside h100-01" >&2
  exit 2
fi

OUT=outputs/fe135_cat_t5_flux_v2_seq_h100/fe_055_cat_7_wooden_sculpture/dece_rf_flux/seed_10
MASK=data/flowedit_compatible_135/derived_masks/fe_055_cat_7_wooden_sculpture_fe_055_cat_7_wooden_sculpture_support_expand_r21_b6.png
mkdir -p "$OUT"

"$PROJ/.venv-h100/bin/python" run_edit_flux.py \
  --model-id black-forest-labs/FLUX.1-dev \
  --cache-dir "$PROJ/.cache/huggingface/hub" \
  --local-files-only \
  --sequential-offload \
  --method dece_rf_flux \
  --seed 10 \
  --num-inference-steps 28 \
  --n-max 24 \
  --max-image-size 512 \
  --true-cfg \
  --distilled-guidance 1.0 \
  --src-guidance-scale 1.0 \
  --base-guidance-scale 1.0 \
  --tar-guidance-scale 5.0 \
  --image "$PROJ/_baselines/src/FlowEdit/Data/Images/cat.png" \
  --source-prompt "A small, fluffy kitten sitting in a grassy field. The kitten is positioned in the center of the scene, surrounded by a field. The kitten appears to be looking at something in the field." \
  --prompt "A smooth clean-edged wooden cat sculpture with no fur, no whiskers, and no floating hair strands. A small wooden sculpture of a kitten sitting in a grassy field. Preserve the camera view and background; only the subject material and surface appearance change." \
  --negative-prompt "fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines" \
  --semantic-base-mask "$PROJ/$MASK" \
  --support-control-mode fixed \
  --support-external-mask-role attention \
  --edit-operation replace \
  --support-relation none \
  --mask-layering-mode none \
  --edit-hedit-guidance-scale 1.30 \
  --edit-guidance-scale 0.14 \
  --edit-region-guidance-scale 0.30 \
  --edit-target-guidance-scale 0.26 \
  --edit-source-guidance-scale 0.01 \
  --edit-local-target-prompt "a smooth clean-edged wooden cat sculpture, carved wooden surface, no fur, no whiskers, no floating hair strands" \
  --edit-local-target-guidance-scale 1.05 \
  --edit-local-target-cfg-scale 5.0 \
  --rec-guidance-scale 0.15 \
  --struct-guidance-scale 0.40 \
  --trajectory-preserve-scale 0.10 \
  --beta-max 1.0 \
  --rec-stop-timestep 0.08 \
  --linear-path-t-min 0.05 \
  --adaptive-clean-control \
  --adaptive-edit-target-rms 0.42 \
  --adaptive-rmsgap-mode legacy \
  --adaptive-preserve-drift-budget 0.12 \
  --adaptive-edit-gain 2.0 \
  --adaptive-preserve-gain 1.6 \
  --adaptive-edit-weight-min 0.85 \
  --adaptive-edit-weight-max 1.55 \
  --adaptive-preserve-weight-min 1.0 \
  --adaptive-preserve-weight-max 1.65 \
  --adaptive-projection-scale 0.65 \
  --adaptive-preserve-clean-correction-scale 0.5 \
  --region-target-transport-scale 0.0 \
  --region-target-outside-lock-scale 0.02 \
  --final-postprocess-mode mask_blend \
  --final-mask-blend-scale 1.0 \
  --final-mask-alpha-gamma 1.0 \
  --new-tokens wooden,sculpture \
  --host-tokens kitten \
  --mask-output-dir "$OUT/masks" \
  --output "$OUT/result.png" \
  --metadata-output "$OUT/metadata.json" \
  --stats-output "$OUT/stats.json"

date
echo fe135_cat_t5_flux_v2_seq_h100_complete
