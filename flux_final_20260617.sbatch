#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J flux-final
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_final_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_final_20260617"; mkdir -p "$OUT"; MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$1" --seeds 10 --output-root "$OUT/_b_$1" >/dev/null
  cat "$OUT/_b_$1/$1/dece_rf_flux/seed_10/command.txt"; }
run () { local n="$1"; local task="$2"; shift 2; local od="$OUT/$n"; mkdir -p "$od"; local B; B="$(gen "$task")"
  echo "=== $n : $* ==="; date
  eval "$B $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
KSTRONG="--edit-hedit-guidance-scale 1.30 --edit-local-target-guidance-scale 1.05 --edit-target-guidance-scale 0.26 --rec-guidance-scale 0.15 --adaptive-preserve-gain 1.6 --trajectory-preserve-scale 0.10 --fixed-core-attention-percentile 20"
# T5 knit: full-face coverage via low percentile + strong edit
run knit_white_p20_relief pillow_same_color_cable_knit      "$KSTRONG"
run knit_white_p20_pure   pillow_same_color_cable_knit      "--final-postprocess-mode mask_blend $KSTRONG"
run knit_grey_p20_pure    pillow_same_color_cable_knit_grey "--final-postprocess-mode mask_blend $KSTRONG"
run knit_grey_p20_relief  pillow_same_color_cable_knit_grey "$KSTRONG"
# leaf: moderate (keep green, original size)
run leaf_mid1 tote_leaf "--edit-hedit-guidance-scale 0.90 --edit-local-target-guidance-scale 0.60 --edit-target-guidance-scale 0.20 --rec-guidance-scale 0.32"
run leaf_mid2 tote_leaf "--edit-hedit-guidance-scale 1.00 --edit-local-target-guidance-scale 0.70 --edit-target-guidance-scale 0.18 --rec-guidance-scale 0.28"
# heart: big mask + prompt without "small"
od="$OUT/heart_promptbig"; mkdir -p "$od"; B="$(gen mug_heart)"
HP='The same white ceramic mug in the same studio scene, with exactly one bold solid red heart, larger and clearly visible at about one third of the mug width, centered on the front, and no other markings.'
HL='The same white ceramic mug with exactly one large bold solid red heart centered on the front of the mug, no other markings.'
echo "=== heart_promptbig ==="; date
eval "$B --support-mask $MV/mug_heart_big.png --edit-hedit-guidance-scale 1.2 --edit-local-target-guidance-scale 1.0 --edit-target-guidance-scale 0.24 --rec-guidance-scale 0.18 --target-prompt \"\$HP\" --edit-local-target-prompt \"\$HL\" --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
date; echo flux_final_complete
