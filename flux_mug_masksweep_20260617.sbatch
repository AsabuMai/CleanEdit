#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-mugsweep
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_mugsweep_a100_%j.log

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
OUT="$PROJECT/outputs/flux_mug_masksweep_20260617"
MV="$PROJECT/outputs/flux_mug_maskvariants"

# mug_heart multi-instance: FLUX tiles "small heart" across the broad 0.21 gate.
# Sweep medium edit-gate sizes (dilated SD3 core) to find the one holding ONE
# solid heart with no tiling room. g5.0 fixed, true CFG, overlay off.
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

run "$OUT/dil02_f067" "$MV/mug_edit_dil02.png" "${mug_heart_args[@]}"
run "$OUT/dil03_f084" "$MV/mug_edit_dil03.png" "${mug_heart_args[@]}"
run "$OUT/dil04_f104" "$MV/mug_edit_dil04.png" "${mug_heart_args[@]}"
run "$OUT/dil06_f147" "$MV/mug_edit_dil06.png" "${mug_heart_args[@]}"

date
echo flux_mug_masksweep_complete
