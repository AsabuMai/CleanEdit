#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-tightmask
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_tightmask_a100_%j.log

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

PY="$PROJECT/.venv-h100/bin/python"
OUT="$PROJECT/outputs/flux_tightmask_test_20260617"

# Multi-instance fix test: FLUX fixed-mode uses external mask directly as the edit
# gate. Broad semantic_support (mug 0.206 / tshirt 0.332) -> motif tiles. SD3's
# effective core is the tight operation_v3_edit_mask (mug 0.026 / tshirt 0.049).
# A/B: same args, only --semantic-base-mask differs (broad vs tight). g5.0 fixed.
run () {
  local outdir="$1"; local mask="$2"; shift 2
  mkdir -p "$outdir"
  echo "=== $outdir mask=$mask ==="; date
  "$PY" "$PROJECT/run_edit_flux.py" \
    --model-id black-forest-labs/FLUX.1-dev \
    --cache-dir "$PROJECT/.cache/huggingface/hub" --local-files-only --model-offload \
    --method dece_rf_flux --seed 10 --num-inference-steps 28 --n-max 24 --max-image-size 512 \
    --true-cfg --distilled-guidance 1.0 --tar-guidance-scale 5.0 \
    --semantic-base-mask "$mask" \
    "$@" \
    --output "$outdir/result.png" --metadata-output "$outdir/metadata.json" --stats-output "$outdir/stats.json"
}

MUG_DIR="$PROJECT/outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap/seed_10/masks"
TSH_DIR="$PROJECT/outputs/pretty_matrix/tshirt_star/support_v3_controller_rmsgap/seed_10/masks"

mug_heart_args=(
  --image "$PROJECT/data/pretty_free_candidates/pexels_white_mug_6312107.jpg"
  --source-prompt "A minimalist photo of a plain white ceramic mug on a grey background."
  --prompt "A small flat red heart decal printed on a white ceramic mug."
  --src-guidance-scale 1.0 --base-guidance-scale 1.0
  --support-control-mode fixed --support-external-mask-role attention
  --edit-operation add_decal --support-relation on_surface --mask-layering-mode object_contact
  --edit-hedit-guidance-scale 0.65 --edit-guidance-scale 0.0 --edit-region-guidance-scale 0.0
  --rec-guidance-scale 0.45 --struct-guidance-scale 0.45 --trajectory-preserve-scale 0.3 --beta-max 1.0
  --rec-stop-timestep 0.08 --linear-path-t-min 0.05 --adaptive-clean-control --adaptive-edit-target-rms 0.42
  --adaptive-rmsgap-mode legacy --adaptive-preserve-drift-budget 0.12 --adaptive-edit-gain 2.0 --adaptive-preserve-gain 4.5
  --adaptive-edit-weight-min 0.85 --adaptive-edit-weight-max 1.55 --adaptive-preserve-weight-min 1.0 --adaptive-preserve-weight-max 1.65
  --adaptive-projection-scale 0.65 --adaptive-preserve-clean-correction-scale 0.5
  --region-target-transport-scale 0.0 --region-target-outside-lock-scale 0.0
)

tshirt_star_args=(
  --image "$PROJECT/data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg"
  --source-prompt "A close-up fashion photo of a person wearing a plain white t-shirt and blue jeans, with natural fabric folds and soft studio lighting."
  --prompt "A flat red star printed on the front of a white T-shirt."
  --src-guidance-scale 1.0 --base-guidance-scale 1.0
  --support-control-mode fixed --support-external-mask-role attention
  --edit-operation add_decal --support-relation on_surface --mask-layering-mode object_contact
  --edit-hedit-guidance-scale 0.65 --edit-guidance-scale 0.0 --edit-region-guidance-scale 0.0
  --rec-guidance-scale 0.45 --struct-guidance-scale 0.45 --trajectory-preserve-scale 0.3 --beta-max 1.0
  --rec-stop-timestep 0.08 --linear-path-t-min 0.05 --adaptive-clean-control --adaptive-edit-target-rms 0.42
  --adaptive-rmsgap-mode legacy --adaptive-preserve-drift-budget 0.12 --adaptive-edit-gain 2.0 --adaptive-preserve-gain 4.5
  --adaptive-edit-weight-min 0.85 --adaptive-edit-weight-max 1.55 --adaptive-preserve-weight-min 1.0 --adaptive-preserve-weight-max 1.65
  --adaptive-projection-scale 0.65 --adaptive-preserve-clean-correction-scale 0.5
  --region-target-transport-scale 0.0 --region-target-outside-lock-scale 0.0
)

run "$OUT/mug_heart_broadmask"   "$MUG_DIR/semantic_support.png"        "${mug_heart_args[@]}"
run "$OUT/mug_heart_tightmask"   "$MUG_DIR/operation_v3_edit_mask.png"  "${mug_heart_args[@]}"
run "$OUT/tshirt_star_broadmask" "$TSH_DIR/semantic_support.png"        "${tshirt_star_args[@]}"
run "$OUT/tshirt_star_tightmask" "$TSH_DIR/operation_v3_edit_mask.png"  "${tshirt_star_args[@]}"

date
echo flux_tightmask_test_complete
