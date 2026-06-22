#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-mugstr
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_mugstr_a100_%j.log

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
OUT="$PROJECT/outputs/flux_mug_strengthsweep_20260617"
# Tight gate = SD3 core (0.027) geometrically limits to ONE location (no tiling
# room). Default edit strength rendered only a faint speck -> boost edit strength
# so a single SOLID heart forms, mirroring SD3's strong single-object growth.
MASK="$PROJECT/outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap/seed_10/masks/operation_v3_edit_mask.png"

run () {
  local outdir="$1"; local hedit="$2"; local rms="$3"; local wmax="$4"; local gain="$5"
  mkdir -p "$outdir"
  echo "=== $outdir hedit=$hedit rms=$rms wmax=$wmax gain=$gain ==="; date
  "$PY" "$PROJECT/run_edit_flux.py" \
    --model-id black-forest-labs/FLUX.1-dev \
    --cache-dir "$PROJECT/.cache/huggingface/hub" --local-files-only --model-offload \
    --method dece_rf_flux --seed 10 --num-inference-steps 28 --n-max 24 --max-image-size 512 \
    --true-cfg --distilled-guidance 1.0 --tar-guidance-scale 5.0 \
    --image "$PROJECT/data/pretty_free_candidates/pexels_white_mug_6312107.jpg" \
    --source-prompt "A minimalist photo of a plain white ceramic mug on a grey background." \
    --prompt "A small flat red heart decal printed on a white ceramic mug." \
    --semantic-base-mask "$MASK" \
    --src-guidance-scale 1.0 --base-guidance-scale 1.0 \
    --support-control-mode fixed --support-external-mask-role attention \
    --edit-operation add_decal --support-relation on_surface --mask-layering-mode object_contact \
    --edit-hedit-guidance-scale "$hedit" --edit-guidance-scale 0.0 --edit-region-guidance-scale 0.0 \
    --rec-guidance-scale 0.45 --struct-guidance-scale 0.45 --trajectory-preserve-scale 0.3 --beta-max 1.0 \
    --rec-stop-timestep 0.08 --linear-path-t-min 0.05 --adaptive-clean-control --adaptive-edit-target-rms "$rms" \
    --adaptive-rmsgap-mode legacy --adaptive-preserve-drift-budget 0.12 --adaptive-edit-gain "$gain" --adaptive-preserve-gain 4.5 \
    --adaptive-edit-weight-min 0.85 --adaptive-edit-weight-max "$wmax" --adaptive-preserve-weight-min 1.0 --adaptive-preserve-weight-max 1.65 \
    --adaptive-projection-scale 0.65 --adaptive-preserve-clean-correction-scale 0.5 \
    --region-target-transport-scale 0.0 --region-target-outside-lock-scale 0.0 \
    --output "$outdir/result.png" --metadata-output "$outdir/metadata.json" --stats-output "$outdir/stats.json"
}

#     outdir              hedit rms  wmax gain
run "$OUT/s1_h10"         1.0   0.42 1.55 2.0
run "$OUT/s2_h15"         1.5   0.55 2.00 2.5
run "$OUT/s3_h20"         2.0   0.70 2.50 3.0
run "$OUT/s4_h26"         2.6   0.85 3.00 3.5

date
echo flux_mug_strengthsweep_complete
