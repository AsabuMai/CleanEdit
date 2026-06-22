#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J flux-tshirtsat
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_tshirtsat_%j.log

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

OUT="$PROJECT/outputs/flux_tshirt_satsweep_20260617"
mkdir -p "$OUT"

# Regenerate the FULL faithful tshirt command (runner config + true-CFG) then
# override strength flags at the end (argparse last-wins) to push star saturation
# toward SD3's solid bright red. Base config already gives single centered star.
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

# name        hedit  local  rec   etarget rtransport  [tar]
run_variant v1_mild   "--edit-hedit-guidance-scale 0.85 --edit-local-target-guidance-scale 0.65 --rec-guidance-scale 0.34 --edit-target-guidance-scale 0.10 --region-target-transport-scale 0.16"
run_variant v2_strong "--edit-hedit-guidance-scale 0.98 --edit-local-target-guidance-scale 0.85 --rec-guidance-scale 0.26 --edit-target-guidance-scale 0.16 --region-target-transport-scale 0.22"
run_variant v3_max    "--edit-hedit-guidance-scale 1.15 --edit-local-target-guidance-scale 0.95 --rec-guidance-scale 0.22 --edit-target-guidance-scale 0.20 --region-target-transport-scale 0.26"
run_variant v4_cfg    "--edit-hedit-guidance-scale 0.98 --edit-local-target-guidance-scale 0.85 --rec-guidance-scale 0.26 --edit-target-guidance-scale 0.16 --region-target-transport-scale 0.22 --tar-guidance-scale 6.5"

date
echo flux_tshirt_satsweep_complete
