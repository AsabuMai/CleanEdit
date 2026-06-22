#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -J flux-greyknit
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_greyknit_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo "wrong host"; exit 2; }
OUT="$PROJECT/outputs/flux_greyknit_20260617"; mkdir -p "$OUT"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --dry-run --true-cfg --tar-guidance 5.0 --tasks pillow_same_color_cable_knit_grey --seeds 10 \
  --output-root "$OUT/_base" >/dev/null
BASE="$(cat "$OUT/_base/pillow_same_color_cable_knit_grey/dece_rf_flux/seed_10/command.txt")"
for s in 6.0 10.0; do
  od="$OUT/grey_s${s/./p}"; mkdir -p "$od"
  echo "=== grey scale=$s ==="; date
  eval "$BASE --final-knit-texture-scale $s --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
done
date; echo flux_greyknit_complete
