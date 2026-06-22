#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J flux-refine
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_refine_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_refine_20260617"; mkdir -p "$OUT"; MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$1" --seeds 10 --output-root "$OUT/_b_$1" >/dev/null
  cat "$OUT/_b_$1/$1/dece_rf_flux/seed_10/command.txt"; }
run () { local n="$1"; local task="$2"; shift 2; local od="$OUT/$n"; mkdir -p "$od"; local B; B="$(gen "$task")"
  echo "=== $n : $* ==="; date
  eval "$B $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
KSTRONG="--edit-hedit-guidance-scale 1.30 --edit-local-target-guidance-scale 1.05 --edit-target-guidance-scale 0.26 --rec-guidance-scale 0.15 --adaptive-preserve-gain 1.6 --trajectory-preserve-scale 0.10"
# T5 knit: strong edit, pure-gen (no relief) vs relief+strong
run knit_white_pure  pillow_same_color_cable_knit      "--final-postprocess-mode mask_blend $KSTRONG"
run knit_white_relief pillow_same_color_cable_knit     "$KSTRONG"
run knit_grey_pure   pillow_same_color_cable_knit_grey "--final-postprocess-mode mask_blend $KSTRONG"
run knit_grey_relief pillow_same_color_cable_knit_grey "$KSTRONG"
# recolor realism: lower blend lets original material/texture through
run mug_soft  green_mug_orange_phase2 "--final-recolor-blend-scale 0.55"
run vase_soft yellow_vase_blue_phase2 "--extra-arg=--recolor-full-mask --final-mask-dilate 2 --final-recolor-blend-scale 0.60"
# leaf smaller/milder
run leaf_mild tote_leaf "--edit-hedit-guidance-scale 0.80 --edit-local-target-guidance-scale 0.45 --edit-target-guidance-scale 0.12 --rec-guidance-scale 0.38"
# heart bigger: big mask + strong edit
run heart_bigstrong mug_heart "--support-mask $MV/mug_heart_big.png --edit-hedit-guidance-scale 1.4 --edit-local-target-guidance-scale 1.1 --edit-target-guidance-scale 0.30 --rec-guidance-scale 0.12"
date; echo flux_refine_complete
