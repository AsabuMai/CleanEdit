#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J flux-objboost
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_objboost_%j.log

set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo "wrong host"; exit 2; }

OUT="$PROJECT/outputs/flux_objects_boost_20260617"; mkdir -p "$OUT"

# Pure-generation prominence boost (NO object overlay) for under-powered add tasks.
# white_bowl_orange was weakest (hedit 0.58 / edit_target 0.05 / local 0.36).
gen_base () {
  local task="$1"
  "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
    --dry-run --true-cfg --tar-guidance 5.0 --tasks "$task" --seeds 10 \
    --output-root "$OUT/_base_$task" >/dev/null
  cat "$OUT/_base_$task/$task/dece_rf_flux/seed_10/command.txt"
}
run_one () {
  local name="$1"; local task="$2"; shift 2
  local od="$OUT/$name"; mkdir -p "$od"
  local BASE; BASE="$(gen_base "$task")"
  echo "=== $name task=$task : $* ==="; date
  eval "$BASE $* --output $od/result.png --metadata-output $od/metadata.json --stats-output $od/stats.json"
}

run_one orange_v1 white_bowl_orange_tabletop_phase2 "--edit-hedit-guidance-scale 0.85 --edit-local-target-guidance-scale 0.75 --edit-target-guidance-scale 0.14 --rec-guidance-scale 0.26"
run_one orange_v2 white_bowl_orange_tabletop_phase2 "--edit-hedit-guidance-scale 1.00 --edit-local-target-guidance-scale 0.90 --edit-target-guidance-scale 0.20 --rec-guidance-scale 0.22"
run_one cat_v1    cat_crown                          "--edit-hedit-guidance-scale 1.10 --edit-local-target-guidance-scale 1.00 --edit-target-guidance-scale 0.22 --rec-guidance-scale 0.14"
run_one bow_v1    dog_bow_tie_phase2                  "--edit-hedit-guidance-scale 1.10 --edit-local-target-guidance-scale 1.10 --edit-target-guidance-scale 0.24 --rec-guidance-scale 0.24"

date
echo flux_objects_boost_complete
