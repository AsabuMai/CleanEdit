#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-orangefix
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_orangefix_%j.log

set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo "wrong host"; exit 2; }

OUT="$PROJECT/outputs/flux_orange_maskfix_20260617"; mkdir -p "$OUT"
MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"

# Floating orange fix: override the broad semantic_support with a custom oval
# grounded ON the wood tabletop, left of the bowl. Keep boosted strength so the
# orange forms solid (matrix config had hedit 0.58 / edit_target 0.05, too weak).
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --dry-run --true-cfg --tar-guidance 5.0 \
  --tasks white_bowl_orange_tabletop_phase2 --seeds 10 \
  --output-root "$OUT/_base" >/dev/null
BASE="$(cat "$OUT/_base/white_bowl_orange_tabletop_phase2/dece_rf_flux/seed_10/command.txt")"

BOOST="--edit-hedit-guidance-scale 1.00 --edit-local-target-guidance-scale 0.90 --edit-target-guidance-scale 0.20 --rec-guidance-scale 0.22"

run_variant () {
  local name="$1"; local mask="$2"
  local od="$OUT/$name"; mkdir -p "$od"
  echo "=== $name mask=$mask ==="; date
  eval "$BASE --support-mask $mask $BOOST \
    --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
}

run_variant v1 "$MV/white_bowl_orange_table_v1.png"
run_variant v2 "$MV/white_bowl_orange_table_v2.png"
run_variant v3 "$MV/white_bowl_orange_table_v3.png"

date
echo flux_orange_maskfix_complete
