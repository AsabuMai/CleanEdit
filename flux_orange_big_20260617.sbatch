#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -J flux-orangebig
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_orangebig_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_orange_big_20260617"; mkdir -p "$OUT"
MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --dry-run --true-cfg --tar-guidance 5.0 --tasks white_bowl_orange_tabletop_phase2 --seeds 10 \
  --output-root "$OUT/_base" >/dev/null
BASE="$(cat "$OUT/_base/white_bowl_orange_tabletop_phase2/dece_rf_flux/seed_10/command.txt")"
BOOST="--edit-hedit-guidance-scale 1.05 --edit-local-target-guidance-scale 0.95 --edit-target-guidance-scale 0.22 --rec-guidance-scale 0.22"
rv () { local n="$1"; local m="$2"; local od="$OUT/$n"; mkdir -p "$od"; echo "=== $n $m ==="; date
  eval "$BASE --support-mask $m $BOOST --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
rv big1 "$MV/white_bowl_orange_table_big1.png"
rv big2 "$MV/white_bowl_orange_table_big2.png"
date; echo flux_orange_big_complete
