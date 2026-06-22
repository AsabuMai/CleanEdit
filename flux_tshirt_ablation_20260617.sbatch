#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J flux-tshirtabl
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_tshirtabl_%j.log

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

OUT="$PROJECT/outputs/flux_tshirt_ablation_20260617"
mkdir -p "$OUT"

# Pale star + ghost double-star appears ONLY with full runner config (not with
# stripped broad-mask config which gave a solid single star). Ablate each added
# mechanism to find the cause of paleness/ghost. Base = runner + true-CFG g5.0.
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --dry-run --true-cfg --tar-guidance 5.0 --tasks tshirt_star --seeds 10 \
  --output-root "$OUT/_base" >/dev/null
BASE_CMD="$(cat "$OUT/_base/tshirt_star/dece_rf_flux/seed_10/command.txt")"

run_variant () {
  local name="$1"; shift
  local od="$OUT/$name"; mkdir -p "$od"
  echo "=== variant $name : $* ==="; date
  eval "$BASE_CMD $* \
    --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
}

run_variant ab1_noregion       "--edit-region-guidance-scale 0.0"
run_variant ab2_noproxytarget  "--edit-target-guidance-scale 0.0"
run_variant ab3_nolocaltransp  "--edit-local-target-guidance-scale 0.0 --region-target-transport-scale 0.0"
run_variant ab4_region_target0 "--edit-region-guidance-scale 0.0 --edit-target-guidance-scale 0.0 --edit-hedit-guidance-scale 0.95 --rec-guidance-scale 0.30"

date
echo flux_tshirt_ablation_complete
