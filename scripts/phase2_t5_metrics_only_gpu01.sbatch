#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J phase2-t5-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_t5_metrics_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_t5_metrics_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
TASKS="pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"
SEEDS="10 11 12"
INTERNAL_METHODS="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap support_v3_fixed"
BASELINES="flowedit flowalign splitflow fireflow rf_solver_edit reflex"

export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run T5 metrics outside a100-01" >&2
  exit 2
fi

echo "== T5 eval assets =="
.venv/bin/python scripts/prepare_t5_eval_assets.py
.venv/bin/python scripts/prepare_e2_t5_baseline_matrix.py

echo "== T5 internal metrics =="
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/table2_t5_internal_metrics.csv" \
  --json-output "$EXP/table2_t5_internal_metrics.json" \
  --task-names "$TASKS" \
  --method-names "$INTERNAL_METHODS" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --allow-download

echo "== T5 baseline metrics =="
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/e2_t5_baseline_matrix" \
  --csv-output "$EXP/table2_t5_baseline_metrics.csv" \
  --json-output "$EXP/table2_t5_baseline_metrics.json" \
  --task-names "$TASKS" \
  --method-names "$BASELINES" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --allow-download

echo "== T5 coverage =="
.venv/bin/python - <<'PY'
import csv
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
for name in ["table2_t5_internal_metrics.csv", "table2_t5_baseline_metrics.csv"]:
    rows = list(csv.DictReader((exp / name).open(newline="", encoding="utf-8")))
    print(name, "rows", len(rows), "tasks", sorted({r["task"] for r in rows}), "methods", sorted({r["method"] for r in rows}))
PY
