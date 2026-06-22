#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH --time=04:00:00
#SBATCH -J t5-reflex-h100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_t5_reflex_h100_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/phase2_t5_reflex_h100_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
TASKS="pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"
SEEDS="10 11 12"
BASELINES="flowedit flowalign splitflow fireflow rf_solver_edit reflex"
MANIFEST="$EXP/e2_t5_formal_baseline_manifest.csv"

export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export REFLEX_LOCAL_FILES_ONLY="${REFLEX_LOCAL_FILES_ONLY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run Phase2 T5 ReFlex retry outside h100-01" >&2
  exit 2
fi

test -f "$MANIFEST"
test -d "$HUGGINGFACE_HUB_CACHE/models--black-forest-labs--FLUX.1-dev/snapshots"

echo "== ReFlex T5 retry =="
_baselines/envs/reflex-py310/bin/python scripts/archive_legacy_2026-05-11/run_reflex_baseline.py \
  --manifest "$MANIFEST" --reflex-root "$PROJECT/_baselines/src/ReFlex" \
  --python "$PROJECT/_baselines/envs/reflex-py310/bin/python" \
  --tasks "$TASKS" --seeds "$SEEDS" --skip-complete

echo "== T5 baseline matrix and metrics =="
.venv-h100/bin/python scripts/prepare_e2_t5_baseline_matrix.py
.venv-h100/bin/python scripts/evaluate_paper_metrics.py \
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

echo "== T5 ReFlex coverage =="
.venv-h100/bin/python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
manifest = list(csv.DictReader((exp / "e2_t5_formal_baseline_manifest.csv").open(newline="", encoding="utf-8")))
print("baseline_manifest_status", dict(Counter(row["status"] for row in manifest)))
rows = list(csv.DictReader((exp / "table2_t5_baseline_metrics.csv").open(newline="", encoding="utf-8")))
print("table2_t5_baseline_metrics.csv rows", len(rows), "methods", sorted({r.get("method", "") for r in rows}))
missing = exp / "e2_t5_baseline_matrix_missing.csv"
print("missing_rows", max(0, sum(1 for _ in missing.open(encoding="utf-8")) - 1) if missing.exists() else 0)
PY
