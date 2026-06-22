#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH -J flux-fixable
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_fixable_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_fixable_20260617"; mkdir -p "$OUT"; MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$1" --seeds 10 --output-root "$OUT/_b_$1" >/dev/null
  cat "$OUT/_b_$1/$1/dece_rf_flux/seed_10/command.txt"; }
run () { local n="$1"; local task="$2"; shift 2; local od="$OUT/$n"; mkdir -p "$od"; local B; B="$(gen "$task")"
  echo "=== $n : $* ==="; date
  eval "$B $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
BOOST="--edit-hedit-guidance-scale 1.1 --edit-local-target-guidance-scale 1.0 --edit-target-guidance-scale 0.22 --rec-guidance-scale 0.22"
run vase_d4    yellow_vase_blue_phase2 "--final-mask-dilate 4"
run vase_d8    yellow_vase_blue_phase2 "--final-mask-dilate 8 --final-recolor-blend-scale 0.92"
run heart_big  mug_heart "--support-mask $MV/mug_heart_big.png $BOOST"
run crown_big  cat_crown "--support-mask $MV/cat_crown_big.png $BOOST"
run leaf_boost tote_leaf "$BOOST"
run leaf_boost2 tote_leaf "--edit-hedit-guidance-scale 1.3 --edit-local-target-guidance-scale 1.1 --edit-target-guidance-scale 0.26 --rec-guidance-scale 0.18"
date; echo flux_fixable_complete
