#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -J flux-vasefix
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_vasefix_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_vasefix_20260617"; mkdir -p "$OUT"; MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks yellow_vase_blue_phase2 --seeds 10 --output-root "$OUT/_b" >/dev/null
  cat "$OUT/_b/yellow_vase_blue_phase2/dece_rf_flux/seed_10/command.txt"; }
run () { local n="$1"; shift; local od="$OUT/$n"; mkdir -p "$od"; local B; B="$(gen)"
  echo "=== $n : $* ==="; date
  eval "$B --support-mask $MV/yellow_vase_full.png $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
run full_d2 "--final-mask-dilate 2 --final-recolor-blend-scale 0.90"
run full_d5 "--final-mask-dilate 5 --final-recolor-blend-scale 0.96 --recolor-clean-projection-scale 0.9"
date; echo flux_vasefix_complete
