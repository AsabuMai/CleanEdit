#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J p2-sel-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_selected_metric_refresh_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_selected_metric_refresh_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"
SEEDS="10 11 12"

export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

echo "== selected-run mirror =="
.venv/bin/python scripts/prepare_phase2_selected_metric_refresh.py

echo "== selected-run metrics =="
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/phase2_selected_metric_refresh_20260612" \
  --csv-output "$EXP/phase2_selected_metric_refresh_metrics.csv" \
  --json-output "$EXP/phase2_selected_metric_refresh_metrics.json" \
  --task-names "$TASKS" \
  --method-names "support_v3_controller_rmsgap" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31

echo "== merge selected metrics =="
.venv/bin/python scripts/merge_phase2_selected_metric_refresh.py

echo "== rebuild paper artifacts =="
.venv/bin/python scripts/build_phase2_paper_tables.py
.venv/bin/python scripts/build_phase2_final_selected_runs.py
.venv/bin/python scripts/build_phase2_main_figure_grid.py
.venv/bin/python scripts/make_phase2_appendix_grids.py
.venv/bin/python scripts/make_phase2_external_baseline_bar_charts.py
.venv/bin/python scripts/build_phase2_docx.py

echo "== final audit =="
.venv/bin/python - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
registry = list(csv.DictReader((exp / "phase2_final_selected_runs_2026-06-11.csv").open(newline="", encoding="utf-8")))
print("registry_rows", len(registry))
print("metric_status", Counter(row.get("metric_status", "") for row in registry))
for name in ["table1_phase2_t1_t5_main_final.csv", "table2a_phase2_sd3_common_subset_final.csv", "table2b_phase2_native_context_final.csv"]:
    rows = list(csv.DictReader((exp / name).open(newline="", encoding="utf-8")))
    print(name, "rows", len(rows), "labels", [row.get("label") for row in rows])
audit = json.loads((exp / "phase2_tables_audit_2026-06-11.json").read_text())
print("table_audit_status", audit.get("status"))
PY

date
echo phase2_selected_metric_refresh_complete
