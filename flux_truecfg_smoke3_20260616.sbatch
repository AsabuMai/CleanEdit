#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH --time=03:00:00
#SBATCH -J flux-truecfg-smoke3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_truecfg_smoke3_%j.log

set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export HF_HOME="$PROJECT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PROJECT/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"
mkdir -p logs

host="$(hostname)"
echo "host=$host"; date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2; exit 2
fi

PY="$PROJECT/.venv/bin/python"
OUT="$PROJECT/outputs/flux_truecfg_smoke_20260616"
IMG="$PROJECT/data/pretty_free_candidates/commons_dog_front_medusa.jpg"
MASK="$PROJECT/outputs/pretty_matrix/dog_front_sunglasses_phase2/support_v3_controller_rmsgap/seed_10/masks/semantic_support.png"
SRC="A close-up front-facing portrait of a dog indoors."
TGT="Black sunglasses aligned across both eyes of a front-facing dog."

# Lean validation: 384px, sequential offload, single CFG scale, NO local-target
# prompt (its guidance scale is 0 so it is unused). Overlay removed -> pure
# DeCE-RF velocity output. Goal: confirm true CFG forms real black sunglasses.
run_one () {
  local name="$1"; shift
  local d="$OUT/$name"
  mkdir -p "$d"
  echo "=== run $name ==="; date
  "$PY" "$PROJECT/run_edit_flux.py" \
    --model-id black-forest-labs/FLUX.1-dev \
    --cache-dir "$PROJECT/.cache/huggingface/hub" --local-files-only --sequential-offload \
    --image "$IMG" --source-prompt "$SRC" --prompt "$TGT" \
    --method dece_rf_flux --seed 10 --num-inference-steps 28 --n-max 24 --max-image-size 384 \
    --src-guidance-scale 1.0 --base-guidance-scale 1.0 \
    --semantic-base-mask "$MASK" --support-control-mode fixed --support-external-mask-role attention \
    --edit-operation add_object --support-relation face_accessory --mask-layering-mode object_contact \
    --edit-hedit-guidance-scale 0.65 --edit-guidance-scale 0.0 --edit-region-guidance-scale 0.0 \
    --rec-guidance-scale 0.2 --struct-guidance-scale 0.45 --trajectory-preserve-scale 0.12 --beta-max 1.0 \
    --rec-stop-timestep 0.08 --linear-path-t-min 0.05 --adaptive-clean-control --adaptive-edit-target-rms 0.42 \
    --adaptive-rmsgap-mode legacy --adaptive-preserve-drift-budget 0.18 --adaptive-edit-gain 2.0 --adaptive-preserve-gain 2.5 \
    --adaptive-edit-weight-min 0.85 --adaptive-edit-weight-max 1.55 --adaptive-preserve-weight-min 1.0 --adaptive-preserve-weight-max 1.65 \
    --adaptive-projection-scale 0.65 --adaptive-preserve-clean-correction-scale 0.5 \
    --region-target-transport-scale 0.0 --region-target-outside-lock-scale 0.0 \
    --true-cfg --distilled-guidance 1.0 "$@" \
    --output "$d/result.png" \
    --metadata-output "$d/metadata.json" --stats-output "$d/stats.json"
}

# Also re-run the control at 384px for a fair same-resolution comparison.
run_ctrl () {
  local d="$OUT/oldmech_noverlay_384"
  mkdir -p "$d"
  echo "=== run oldmech_noverlay_384 ==="; date
  "$PY" "$PROJECT/run_edit_flux.py" \
    --model-id black-forest-labs/FLUX.1-dev \
    --cache-dir "$PROJECT/.cache/huggingface/hub" --local-files-only --sequential-offload \
    --image "$IMG" --source-prompt "$SRC" --prompt "$TGT" \
    --method dece_rf_flux --seed 10 --num-inference-steps 28 --n-max 24 --max-image-size 384 \
    --src-guidance-scale 1.0 --base-guidance-scale 1.0 --tar-guidance-scale 10.5 \
    --semantic-base-mask "$MASK" --support-control-mode fixed --support-external-mask-role attention \
    --edit-operation add_object --support-relation face_accessory --mask-layering-mode object_contact \
    --edit-hedit-guidance-scale 0.65 --edit-guidance-scale 0.0 --edit-region-guidance-scale 0.0 \
    --rec-guidance-scale 0.2 --struct-guidance-scale 0.45 --trajectory-preserve-scale 0.12 --beta-max 1.0 \
    --rec-stop-timestep 0.08 --linear-path-t-min 0.05 --adaptive-clean-control --adaptive-edit-target-rms 0.42 \
    --adaptive-rmsgap-mode legacy --adaptive-preserve-drift-budget 0.18 --adaptive-edit-gain 2.0 --adaptive-preserve-gain 2.5 \
    --adaptive-edit-weight-min 0.85 --adaptive-edit-weight-max 1.55 --adaptive-preserve-weight-min 1.0 --adaptive-preserve-weight-max 1.65 \
    --adaptive-projection-scale 0.65 --adaptive-preserve-clean-correction-scale 0.5 \
    --region-target-transport-scale 0.0 --region-target-outside-lock-scale 0.0 \
    --output "$d/result.png" \
    --metadata-output "$d/metadata.json" --stats-output "$d/stats.json"
}

run_ctrl
run_one truecfg_g5p0_384 --tar-guidance-scale 5.0

date
echo flux_truecfg_smoke3_complete
