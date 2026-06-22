#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-nocomp
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_nocomp_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_nocomposite_20260617"; mkdir -p "$OUT"
# Match SD3: disable the final hard source-paste composite (--final-mask-blend-scale 0)
# so FLUX returns the raw diffusion output (with natural shadows/integration),
# relying on in-diffusion DeCE-RF preservation like SD3 does.
run_one () {
  local task="$1"; local od="$OUT/$task"; mkdir -p "$od"
  "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$task" --seeds 10 \
    --output-root "$OUT/_base_$task" >/dev/null
  local BASE; BASE="$(cat "$OUT/_base_$task/$task/dece_rf_flux/seed_10/command.txt")"
  echo "=== $task (no composite) ==="; date
  eval "$BASE --final-mask-blend-scale 0 \
    --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
}
run_one white_bowl_orange_tabletop_phase2
run_one cat_crown
run_one tshirt_star
run_one mug_heart
date; echo flux_nocomposite_complete
