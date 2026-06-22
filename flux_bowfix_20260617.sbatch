#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -J flux-bowfix
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_bowfix_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_bowfix_20260617"; mkdir -p "$OUT"
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks dog_bow_tie_phase2 --seeds 10 --output-root "$OUT/_b" >/dev/null
  cat "$OUT/_b/dog_bow_tie_phase2/dece_rf_flux/seed_10/command.txt"; }
run () { local n="$1"; shift; local od="$OUT/$n"; mkdir -p "$od"; local B; B="$(gen)"
  echo "=== $n : $* ==="; date
  eval "$B $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
# Kill edit_region + region_transport (source of the chest red block beyond core),
# lower percentile for bow structure.
B="--edit-hedit-guidance-scale 1.00 --edit-local-target-guidance-scale 1.00 --edit-target-guidance-scale 0.18 --edit-region-guidance-scale 0.0 --region-target-transport-scale 0.0"
run bx1_p30 "$B --fixed-core-attention-percentile 30"
run bx2_p20 "$B --fixed-core-attention-percentile 20"
run bx3_p45 "$B --fixed-core-attention-percentile 45"
run bx4_p30_rec "$B --fixed-core-attention-percentile 30 --rec-guidance-scale 0.45"
date; echo flux_bowfix_complete
