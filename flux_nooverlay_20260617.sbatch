#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -J flux-nooverlay
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_nooverlay_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_nooverlay_20260617"; mkdir -p "$OUT"; MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$1" --seeds 10 --output-root "$OUT/_b_$1" >/dev/null
  cat "$OUT/_b_$1/$1/dece_rf_flux/seed_10/command.txt"; }
run () { local n="$1"; local task="$2"; shift 2; local od="$OUT/$n"; mkdir -p "$od"; local B; B="$(gen "$task")"
  echo "=== $n : $* ==="; date
  # --final-postprocess-mode mask_blend -> disables the hand-drawn overlay (pure generation)
  eval "$B --final-postprocess-mode mask_blend $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
run bow_puregen  dog_bow_tie_phase2 "--support-mask $MV/dog_bow_tie_wide.png --fixed-core-attention-percentile 25 --edit-region-guidance-scale 0.0 --region-target-transport-scale 0.0 --edit-hedit-guidance-scale 1.0 --edit-local-target-guidance-scale 1.0 --edit-target-guidance-scale 0.18"
run sg_puregen   dog_front_sunglasses_phase2 "--fixed-core-attention-percentile 20 --edit-hedit-guidance-scale 1.0 --edit-local-target-guidance-scale 1.0 --edit-target-guidance-scale 0.18"
run bow_ov_off_origmask dog_bow_tie_phase2 "--fixed-core-attention-percentile 25 --edit-region-guidance-scale 0.0 --region-target-transport-scale 0.0 --edit-hedit-guidance-scale 1.0 --edit-local-target-guidance-scale 1.0 --edit-target-guidance-scale 0.18"
date; echo flux_nooverlay_complete
