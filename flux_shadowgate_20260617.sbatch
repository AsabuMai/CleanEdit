#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-shadowgate
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_shadowgate_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_shadowgate_20260617"; mkdir -p "$OUT"
MV="$PROJECT/outputs/phase2_dece_flux_custom_masks_20260616"

# Enlarged edit gate (incl. contact/shadow band) + loosened preserve (lower
# rec/traj/preserve_gain, outside_lock 0) + no hard composite -> let diffusion
# paint cast shadow / ground contact around 3D objects.
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$1" --seeds 10 \
    --output-root "$OUT/_base_$1" >/dev/null
  cat "$OUT/_base_$1/$1/dece_rf_flux/seed_10/command.txt"; }

run () { local name="$1"; local task="$2"; local mask="$3"; shift 3
  local od="$OUT/$name"; mkdir -p "$od"; local BASE; BASE="$(gen "$task")"
  echo "=== $name mask=$mask : $* ==="; date
  eval "$BASE --support-mask $mask --final-mask-blend-scale 0 $* \
    --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }

OB="--edit-hedit-guidance-scale 1.00 --edit-local-target-guidance-scale 0.90 --edit-target-guidance-scale 0.20"
run orange_a white_bowl_orange_tabletop_phase2 "$MV/wbo_shadow1.png" "$OB --rec-guidance-scale 0.12 --trajectory-preserve-scale 0.08 --adaptive-preserve-gain 1.8 --region-target-outside-lock-scale 0.0"
run orange_b white_bowl_orange_tabletop_phase2 "$MV/wbo_shadow2.png" "$OB --rec-guidance-scale 0.08 --trajectory-preserve-scale 0.05 --adaptive-preserve-gain 1.4 --region-target-outside-lock-scale 0.0"
run orange_c white_bowl_orange_tabletop_phase2 "$MV/wbo_shadow1.png" "$OB --rec-guidance-scale 0.18 --trajectory-preserve-scale 0.12 --adaptive-preserve-gain 2.2 --region-target-outside-lock-scale 0.0"
run cat_a    cat_crown                          "$MV/cat_crown_shadowgate.png" "--edit-hedit-guidance-scale 1.0 --edit-local-target-guidance-scale 0.95 --edit-target-guidance-scale 0.18 --rec-guidance-scale 0.10 --trajectory-preserve-scale 0.06 --adaptive-preserve-gain 1.6 --region-target-outside-lock-scale 0.0"
date; echo flux_shadowgate_complete
