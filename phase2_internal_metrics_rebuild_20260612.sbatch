#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=03:00:00
#SBATCH -J phase2-metric-v3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_metric_v3_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_metric_v3_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2 pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"
INTERNAL_METHODS="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap support_v3_fixed"
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

echo "== deps =="
.venv/bin/python - <<'PY'
import docx
import lpips
print("deps ok")
PY

echo "== internal metrics with BG/LPIPS/CLIP-T =="
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/phase2_internal_bg_metrics.csv" \
  --json-output "$EXP/phase2_internal_bg_metrics.json" \
  --task-names "$TASKS" \
  --method-names "$INTERNAL_METHODS" \
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
.venv/bin/python scripts/make_phase2_integrated_seed10_audit.py
.venv/bin/python scripts/build_phase2_docx.py

echo "== coverage =="
.venv/bin/python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
for name in [
    "phase2_internal_bg_metrics.csv",
    "e2_t1_t4_baseline_fixed_mask_metrics.csv",
    "table2_t5_baseline_metrics.csv",
]:
    rows = list(csv.DictReader((exp / name).open(newline="", encoding="utf-8")))
    fields = rows[0].keys() if rows else []
    wanted = ["outside_l1", "bg_psnr", "bg_lpips_x100", "bg_ssim_luma", "clip_t", "local_clip_t", "edit_score", "inside_l1"]
    print(name, "rows", len(rows), {key: key in fields for key in wanted})
for name in [
    "table1_phase2_t1_t5_main_final.csv",
    "table2a_phase2_sd3_common_subset_final.csv",
    "table2b_phase2_native_context_final.csv",
]:
    rows = list(csv.DictReader((exp / name).open(newline="", encoding="utf-8")))
    print(name, "rows", len(rows), "methods", [row["method"] for row in rows])
PY

date
echo phase2_metric_v3_complete
