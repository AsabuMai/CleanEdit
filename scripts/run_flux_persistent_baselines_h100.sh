#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J flux-persist
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/flux_persistent_h100_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/flux_persistent_h100_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
MANIFEST=${MANIFEST:-$PROJECT/experiments/support_v3_2026-06-02/e2_t5_formal_baseline_manifest.csv}
TASKS=${TASKS:-}
SEEDS=${SEEDS:-10 11 12}
PERSIST_BASELINES=${PERSIST_BASELINES:-fireflow rf_solver_edit}
LIMIT=${LIMIT:-0}
SKIP_COMPLETE=${SKIP_COMPLETE:-1}
OVERWRITE=${OVERWRITE:-0}

export HF_HOME=${HF_HOME:-$PROJECT/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DISABLE_BASELINE_NSFW=${DISABLE_BASELINE_NSFW:-1}

mkdir -p "$PROJECT/_baselines/logs/generation_matrix"
cd "$PROJECT"

host=$(hostname)
echo "host=$host"
if [[ "$host" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run persistent FLUX baselines outside h100-01" >&2
  exit 2
fi

common_args=(
  --manifest "$MANIFEST"
  --seeds "$SEEDS"
)
if [[ -n "$TASKS" ]]; then
  common_args+=(--tasks "$TASKS")
fi
if [[ "$LIMIT" != "0" ]]; then
  common_args+=(--limit "$LIMIT")
fi
if [[ "$SKIP_COMPLETE" == "1" ]]; then
  common_args+=(--skip-complete)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  common_args+=(--overwrite)
fi

for baseline in $PERSIST_BASELINES; do
  case "$baseline" in
    fireflow)
      echo "== persistent fireflow =="
      _baselines/envs/fireflow-py310/bin/python scripts/run_flux_persistent_baselines.py \
        "${common_args[@]}" \
        --baselines fireflow
      ;;
    rf_solver_edit)
      echo "== persistent rf_solver_edit =="
      _baselines/envs/rf-solver-edit-py310/bin/python scripts/run_flux_persistent_baselines.py \
        "${common_args[@]}" \
        --baselines rf_solver_edit
      ;;
    *)
      echo "Unknown persistent FLUX baseline: $baseline" >&2
      exit 2
      ;;
  esac
done

echo "persistent_flux_baselines_done"
