#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=36:00:00
#SBATCH -J phase2-sam-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_samflow_sd3_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_samflow_sd3_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
EXP="$PROJECT/experiments/support_v3_2026-06-02"
TASKS_T1_T4="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"
TASKS_T5="pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"
SEEDS="10 11 12"
SAM_BASELINES="sam_flow_sd3"

export HF_HOME="$PROJECT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PROJECT/.cache/huggingface/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT"
mkdir -p "$SRC" "$BASE/logs"

host="$(hostname)"
echo "host=$host"
date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

fetch_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local path="$SRC/$name"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi
  git -C "$path" fetch --all --tags --prune
  git -C "$path" checkout "$commit"
  git -C "$path" rev-parse HEAD
}

echo "== fetch Sam-Flow =="
fetch_repo "Sam-Flow" "https://github.com/chwbob/Sam-Flow.git" "f5f70c9dcc77b0dc6b90c29bb863452128850cb5"

echo "== prepare Sam-Flow env =="
python3 scripts/prepare_e2_baseline_envs.py --only Sam-Flow --timeout-s 3600

echo "== smoke Sam-Flow entrypoint =="
python3 scripts/smoke_baseline_entrypoints.py --only Sam-Flow || true

echo "== ensure Sam-Flow manifest rows =="
.venv/bin/python scripts/ensure_samflow_manifest_rows.py --scope all --baselines "$SAM_BASELINES" --seeds "$SEEDS"

echo "== run Sam-Flow T1-T4 =="
"$BASE/envs/sam-flow-py310/bin/python" scripts/run_samflow_baseline.py \
  --manifest "$EXP/e2_t1_t4_formal_baseline_manifest.csv" \
  --samflow-root "$SRC/Sam-Flow" \
  --python "$BASE/envs/sam-flow-py310/bin/python" \
  --baselines "$SAM_BASELINES" \
  --tasks "$TASKS_T1_T4" \
  --seeds "$SEEDS" \
  --max-image-size 512 \
  --skip-complete

echo "== run Sam-Flow T5 =="
"$BASE/envs/sam-flow-py310/bin/python" scripts/run_samflow_baseline.py \
  --manifest "$EXP/e2_t5_formal_baseline_manifest.csv" \
  --samflow-root "$SRC/Sam-Flow" \
  --python "$BASE/envs/sam-flow-py310/bin/python" \
  --baselines "$SAM_BASELINES" \
  --tasks "$TASKS_T5" \
  --seeds "$SEEDS" \
  --max-image-size 512 \
  --skip-complete

echo "== mirror matrices =="
.venv/bin/python scripts/prepare_e2_t1_t4_baseline_matrix.py
.venv/bin/python scripts/prepare_e2_t5_baseline_matrix.py

echo "== metrics T1-T4 =="
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/e2_t1_t4_baseline_matrix" \
  --csv-output "$EXP/e2_t1_t4_baseline_fixed_mask_metrics.csv" \
  --json-output "$EXP/e2_t1_t4_baseline_fixed_mask_metrics.json" \
  --task-names "$TASKS_T1_T4" \
  --method-names "flowedit flowalign splitflow sam_flow_sd3 fireflow rf_solver_edit reflex sam_flow_flux" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/e4_t1_t4_reconstruction_floor_metrics.csv"

echo "== metrics T5 =="
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/e2_t5_baseline_matrix" \
  --csv-output "$EXP/table2_t5_baseline_metrics.csv" \
  --json-output "$EXP/table2_t5_baseline_metrics.json" \
  --task-names "$TASKS_T5" \
  --method-names "flowedit flowalign splitflow sam_flow_sd3 fireflow rf_solver_edit reflex sam_flow_flux" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --allow-download

echo "== rebuild paper artifacts =="
.venv/bin/python scripts/build_phase2_paper_tables.py
.venv/bin/python scripts/build_phase2_final_selected_runs.py
.venv/bin/python scripts/build_phase2_main_figure_grid.py
.venv/bin/python scripts/make_phase2_appendix_grids.py

echo "== coverage =="
.venv/bin/python - <<'PY'
import csv
from collections import Counter, defaultdict
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
for name in ["e2_t1_t4_formal_baseline_manifest.csv", "e2_t5_formal_baseline_manifest.csv"]:
    rows = list(csv.DictReader((exp / name).open(newline="", encoding="utf-8")))
    by = defaultdict(Counter)
    for row in rows:
        if row["baseline"].startswith("sam_flow"):
            by[row["baseline"]][row["status"]] += 1
    print(name, {k: dict(v) for k, v in by.items()})
for name in ["e2_t1_t4_baseline_fixed_mask_metrics.csv", "table2_t5_baseline_metrics.csv"]:
    rows = list(csv.DictReader((exp / name).open(newline="", encoding="utf-8")))
    counts = Counter(row.get("method", "") for row in rows if row.get("method", "").startswith("sam_flow"))
    print(name, dict(counts))
PY

date
echo phase2_samflow_complete
