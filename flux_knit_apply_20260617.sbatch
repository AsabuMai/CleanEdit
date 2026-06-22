#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-knitapply
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_knitapply_%j.log

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

OUT="$PROJECT/outputs/flux_knit_apply_20260617"
mkdir -p "$OUT"

# Apply chosen knit-texture-scale 6.0 to all 3 pillows. Grey has hardcoded
# cable_scale 0.28 (~3.5x weaker), so also try grey at 12.0.
gen_base () {
  local task="$1"
  "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$task" --seeds 10 \
    --output-root "$OUT/_base_$task" >/dev/null
  cat "$OUT/_base_$task/$task/dece_rf_flux/seed_10/command.txt"
}

run_one () {
  local name="$1"; local task="$2"; local scale="$3"
  local od="$OUT/$name"; mkdir -p "$od"
  local BASE; BASE="$(gen_base "$task")"
  echo "=== $name task=$task knit_scale=$scale ==="; date
  eval "$BASE --final-knit-texture-scale $scale \
    --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
}

run_one white_s6    pillow_same_color_cable_knit          6.0
run_one grey_s6     pillow_same_color_cable_knit_grey     6.0
run_one grey_s12    pillow_same_color_cable_knit_grey     12.0
run_one armchair_s6 pillow_same_color_cable_knit_armchair 6.0

date
echo flux_knit_apply_complete
