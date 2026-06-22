#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-structgate
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_structgate_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
OUT="$PROJECT/outputs/flux_structgate_20260617"; mkdir -p "$OUT"
gen () { "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$1" --seeds 10 --output-root "$OUT/_b_$1" >/dev/null
  cat "$OUT/_b_$1/$1/dece_rf_flux/seed_10/command.txt"; }
run () { local n="$1"; local task="$2"; shift 2; local od="$OUT/$n"; mkdir -p "$od"; local B; B="$(gen "$task")"
  echo "=== $n : $* ==="; date
  eval "$B $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"; }
# Lower core-attention-percentile => core ~= full connected band => edit spans
# the whole accessory footprint => connected glasses frame / structured bow tie.
SGB="--edit-hedit-guidance-scale 1.00 --edit-local-target-guidance-scale 1.00 --edit-target-guidance-scale 0.18"
run sg_p20  dog_front_sunglasses_phase2 "$SGB --fixed-core-attention-percentile 20"
run sg_p45  dog_front_sunglasses_phase2 "$SGB --fixed-core-attention-percentile 45"
run bow_p15 dog_bow_tie_phase2          "$SGB --fixed-core-attention-percentile 15"
run bow_p35 dog_bow_tie_phase2          "$SGB --fixed-core-attention-percentile 35"
date; echo flux_structgate_complete
