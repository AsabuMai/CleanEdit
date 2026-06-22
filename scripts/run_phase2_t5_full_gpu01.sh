#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=18:00:00
#SBATCH -J phase2-t5-full
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_t5_full_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_t5_full_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
TASKS="pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"
SEEDS="10 11 12"
INTERNAL_METHODS="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap support_v3_fixed"
BASELINES="flowedit flowalign splitflow sam_flow_sd3 fireflow rf_solver_edit reflex sam_flow_flux"
MANIFEST="$EXP/e2_t5_formal_baseline_manifest.csv"
export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export REFLEX_LOCAL_FILES_ONLY="${REFLEX_LOCAL_FILES_ONLY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run Phase2 T5 full job outside a100-01" >&2
  exit 2
fi

echo "== internal T5 pretty_matrix =="
RUN_ID="phase2_t5_internal_${SLURM_JOB_ID:-manual}" \
TASKS="$TASKS" \
METHODS="$INTERNAL_METHODS" \
SEEDS="$SEEDS" \
SKIP_EXISTING=1 \
BATCH_SKIP_EXISTING=1 \
MODEL_OFFLOAD=0 \
ALLOW_MASK_DOWNLOAD=1 \
RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1 \
bash scripts/run_wacv_phase1_batch.sh

echo "== T5 baseline manifest =="
.venv/bin/python scripts/init_e2_t5_formal_manifest.py \
  --out "$MANIFEST" \
  --tasks "$TASKS" \
  --baselines "$BASELINES" \
  --seeds "$SEEDS"

echo "== SD3 baselines T5 =="
_baselines/envs/flowedit-py310/bin/python scripts/archive_legacy_2026-05-11/run_flowedit_baseline.py \
  --manifest "$MANIFEST" --flowedit-root "$PROJECT/_baselines/src/FlowEdit" \
  --python "$PROJECT/_baselines/envs/flowedit-py310/bin/python" \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete
_baselines/envs/flowalign-py310/bin/python scripts/run_flowalign_baseline.py \
  --manifest "$MANIFEST" --flowalign-root "$PROJECT/_baselines/src/FlowAlign" \
  --python "$PROJECT/_baselines/envs/flowalign-py310/bin/python" \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete
_baselines/envs/splitflow-py310/bin/python scripts/archive_legacy_2026-05-11/run_splitflow_baseline.py \
  --manifest "$MANIFEST" --splitflow-root "$PROJECT/_baselines/src/SplitFlow" \
  --python "$PROJECT/_baselines/envs/splitflow-py310/bin/python" \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete
_baselines/envs/sam-flow-py310/bin/python scripts/run_samflow_baseline.py \
  --manifest "$MANIFEST" --samflow-root "$PROJECT/_baselines/src/Sam-Flow" \
  --python "$PROJECT/_baselines/envs/sam-flow-py310/bin/python" \
  --baselines sam_flow_sd3 \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete

echo "== native FLUX baselines T5 =="
_baselines/envs/fireflow-py310/bin/python scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py \
  --manifest "$MANIFEST" --fireflow-root "$PROJECT/_baselines/src/FireFlow" \
  --python "$PROJECT/_baselines/envs/fireflow-py310/bin/python" \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete
_baselines/envs/rf-solver-edit-py310/bin/python scripts/archive_legacy_2026-05-11/run_rf_solver_edit_baseline.py \
  --manifest "$MANIFEST" --rf-solver-root "$PROJECT/_baselines/src/RF-Solver-Edit/FLUX_Image_Edit" \
  --python "$PROJECT/_baselines/envs/rf-solver-edit-py310/bin/python" \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete
_baselines/envs/reflex-py310/bin/python scripts/archive_legacy_2026-05-11/run_reflex_baseline.py \
  --manifest "$MANIFEST" --reflex-root "$PROJECT/_baselines/src/ReFlex" \
  --python "$PROJECT/_baselines/envs/reflex-py310/bin/python" \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete
_baselines/envs/sam-flow-py310/bin/python scripts/run_samflow_baseline.py \
  --manifest "$MANIFEST" --samflow-root "$PROJECT/_baselines/src/Sam-Flow" \
  --python "$PROJECT/_baselines/envs/sam-flow-py310/bin/python" \
  --baselines sam_flow_flux \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete

echo "== T5 eval assets and metrics =="
.venv/bin/python scripts/prepare_t5_eval_assets.py
.venv/bin/python scripts/prepare_e2_t5_baseline_matrix.py

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
from collections import Counter
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
manifest = list(csv.DictReader((exp / "e2_t5_formal_baseline_manifest.csv").open(newline="", encoding="utf-8")))
counts = Counter(row["status"] for row in manifest)
print("baseline_manifest_status", dict(counts))
for path in [exp / "table2_t5_internal_metrics.csv", exp / "table2_t5_baseline_metrics.csv"]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    print(path.name, "rows", len(rows), "methods", sorted({r.get("method", "") for r in rows}))
PY
