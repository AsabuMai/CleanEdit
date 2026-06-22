#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J flux-tshirtfin
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_tshirtfin_%j.log

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

OUT="$PROJECT/outputs/flux_tshirt_final_20260617"
mkdir -p "$OUT"

# Ghost came from edit_region (broad-ring 2nd edit) + local_target/region_transport.
# Ghost-free base = edit_region 0 + local_target 0 + region_transport 0. Now boost
# saturation toward SD3 solid red via hedit up / rec down.
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --dry-run --true-cfg --tar-guidance 5.0 --tasks tshirt_star --seeds 10 \
  --output-root "$OUT/_base" >/dev/null
BASE_CMD="$(cat "$OUT/_base/tshirt_star/dece_rf_flux/seed_10/command.txt")"

GHOSTFREE="--edit-region-guidance-scale 0.0 --edit-local-target-guidance-scale 0.0 --region-target-transport-scale 0.0"

run_variant () {
  local name="$1"; shift
  local od="$OUT/$name"; mkdir -p "$od"
  echo "=== variant $name : $* ==="; date
  eval "$BASE_CMD $GHOSTFREE $* \
    --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
}

run_variant f1_h95   "--edit-hedit-guidance-scale 0.95 --rec-guidance-scale 0.28"
run_variant f2_h110  "--edit-hedit-guidance-scale 1.10 --rec-guidance-scale 0.24 --edit-target-guidance-scale 0.12"
run_variant f3_h085  "--edit-hedit-guidance-scale 0.85 --rec-guidance-scale 0.34"
run_variant f4_cfg6  "--edit-hedit-guidance-scale 0.98 --rec-guidance-scale 0.26 --tar-guidance-scale 6.0"

date
echo flux_tshirt_final_complete
