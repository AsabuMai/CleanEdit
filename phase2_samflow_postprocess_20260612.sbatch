#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=04:00:00
#SBATCH -J samflow-post
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/samflow_postprocess_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/samflow_postprocess_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
TASKS_T1_T4="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"
TASKS_T5="pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"
SEEDS="10 11 12"
BASELINES="flowedit flowalign splitflow sam_flow_sd3 fireflow rf_solver_edit reflex sam_flow_flux"

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

echo "== metric deps =="
.venv/bin/python - <<'PY' || .venv/bin/python -m pip install lpips==0.1.4 python-docx==1.2.0
import docx
import lpips
print("metric deps ok")
PY

echo "== mirror matrices =="
.venv/bin/python scripts/prepare_e2_t1_t4_baseline_matrix.py
.venv/bin/python scripts/prepare_e2_t5_baseline_matrix.py

echo "== metrics T1-T4 =="
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/e2_t1_t4_baseline_matrix" \
  --csv-output "$EXP/e2_t1_t4_baseline_fixed_mask_metrics.csv" \
  --json-output "$EXP/e2_t1_t4_baseline_fixed_mask_metrics.json" \
  --task-names "$TASKS_T1_T4" \
  --method-names "$BASELINES" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --allow-download

echo "== metrics T5 =="
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/e2_t5_baseline_matrix" \
  --csv-output "$EXP/table2_t5_baseline_metrics.csv" \
  --json-output "$EXP/table2_t5_baseline_metrics.json" \
  --task-names "$TASKS_T5" \
  --method-names "$BASELINES" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --allow-download

echo "== rebuild paper artifacts =="
.venv/bin/python scripts/build_phase2_paper_tables.py
.venv/bin/python scripts/build_phase2_final_selected_runs.py
.venv/bin/python scripts/build_phase2_main_figure_grid.py
.venv/bin/python scripts/make_phase2_appendix_grids.py
.venv/bin/python scripts/make_samflow_seed10_preview.py
if [[ -f scripts/build_phase2_docx.py ]]; then
  .venv/bin/python scripts/build_phase2_docx.py
fi

echo "== coverage =="
.venv/bin/python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
for name in ["e2_t1_t4_formal_baseline_manifest.csv", "e2_t5_formal_baseline_manifest.csv"]:
    rows = list(csv.DictReader((exp / name).open(newline="", encoding="utf-8")))
    counts = Counter((row["baseline"], row["status"]) for row in rows if row["baseline"].startswith("sam_flow"))
    print(name, dict(counts))
for name in ["e2_t1_t4_baseline_fixed_mask_metrics.csv", "table2_t5_baseline_metrics.csv"]:
    rows = list(csv.DictReader((exp / name).open(newline="", encoding="utf-8")))
    counts = Counter(row.get("method", "") for row in rows if row.get("method", "").startswith("sam_flow"))
    print(name, dict(counts))
PY

date
echo samflow_postprocess_complete
