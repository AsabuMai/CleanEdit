#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-knitscale
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_knitscale_%j.log

set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"
mkdir -p logs

host="$(hostname)"
echo "host=$host"; date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2; exit 2
fi

OUT="$PROJECT/outputs/flux_knit_scalesweep_20260617"
mkdir -p "$OUT"

# Knit relief is ALREADY active for material (runner forces debug_visual). At
# final_knit_texture_scale 1.20 it is too faint (grid). relief *= scale (linear),
# so sweep the scale on the white pillow to make cables pronounced like SD3.
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --dry-run --true-cfg --tar-guidance 5.0 \
  --tasks pillow_same_color_cable_knit --seeds 10 \
  --output-root "$OUT/_base" >/dev/null
BASE_CMD="$(cat "$OUT/_base/pillow_same_color_cable_knit/dece_rf_flux/seed_10/command.txt")"

run_variant () {
  local name="$1"; local scale="$2"
  local od="$OUT/$name"; mkdir -p "$od"
  echo "=== variant $name knit_scale=$scale ==="; date
  eval "$BASE_CMD --final-knit-texture-scale $scale \
    --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
}

run_variant s2p5 2.5
run_variant s4p0 4.0
run_variant s6p0 6.0
run_variant s9p0 9.0

date
echo flux_knit_scalesweep_complete
