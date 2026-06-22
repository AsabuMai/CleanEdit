#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -J flux-bowshape
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_bowshape_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_bowshape_20260617"; mkdir -p "$OUT"; MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks dog_bow_tie_phase2 --seeds 10 --output-root "$OUT/_b" >/dev/null
  cat "$OUT/_b/dog_bow_tie_phase2/dece_rf_flux/seed_10/command.txt"; }
run () { local n="$1"; local mask="$2"; shift 2; local od="$OUT/$n"; mkdir -p "$od"; local B; B="$(gen)"
  echo "=== $n mask=$mask ==="; date
  eval "$B --support-mask $mask --edit-region-guidance-scale 0.0 --region-target-transport-scale 0.0 --fixed-core-attention-percentile 25 --edit-hedit-guidance-scale 1.0 --edit-local-target-guidance-scale 1.0 --edit-target-guidance-scale 0.18 $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
run wide  "$MV/dog_bow_tie_wide.png"
run shape "$MV/dog_bow_tie_shape.png"
run shape_p40 "$MV/dog_bow_tie_shape.png" "--fixed-core-attention-percentile 40"
date; echo flux_bowshape_complete
