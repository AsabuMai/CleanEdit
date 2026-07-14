#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH -J t4-tex-metric
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/t4_texture_metrics_%j.log

set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/t4_texture_metrics
RUNS=$PROJ/outputs/t4_texture_metric_runs
MASK_DIR=$PROJ/data/flowedit_compatible_135/eval_masks
LOCAL_PROMPTS=$PROJ/data/flowedit_compatible_135/local_target_prompts.json

export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LOCAL_PROMPTS_JSON=$LOCAL_PROMPTS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

mkdir -p "$EXP"

"$PROJ/.venv/bin/python" scripts/prepare_t4_texture_metric_runs.py

"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$RUNS" \
  --csv-output "$EXP/metrics.csv" \
  --json-output "$EXP/metrics.json" \
  --task-names all \
  --method-names "ours_sd3 ours_flux" \
  --seeds 10 \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --eval-mask-dir "$MASK_DIR"

"$PROJ/.venv/bin/python" scripts/t4_texture_color_metric.py \
  --runs "$RUNS" \
  --eval-mask-dir "$MASK_DIR" \
  --csv-output "$EXP/color_metrics.csv" \
  --summary-output "$EXP/color_summary.csv"

"$PROJ/.venv/bin/python" - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

exp = Path("experiments/t4_texture_metrics")
rows = list(csv.DictReader((exp / "metrics.csv").open(newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    groups[row["method"]].append(row)

metric_cols = [
    "clip_t",
    "clip_target_minus_source",
    "clip_direction_similarity",
    "local_clip_t",
    "local_clip_target_minus_source",
    "bg_lpips",
    "bg_l1",
    "bg_dino_source",
    "bg_ssim_luma",
    "eval_mask_area",
    "bg_mask_area",
]

def vals(items, key):
    out = []
    for row in items:
        raw = row.get(key, "")
        if raw == "":
            continue
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out

summary = []
for method, items in sorted(groups.items()):
    rec = {
        "method": method,
        "n": len(items),
        "complete_n": sum(str(row.get("complete")) == "True" for row in items),
        "fixed_mask_n": sum(row.get("mask_source") == "fixed_eval_mask" for row in items),
    }
    for key in metric_cols:
        xs = vals(items, key)
        rec[key] = "" if not xs else sum(xs) / len(xs)
    summary.append(rec)

with (exp / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["method", "n", "complete_n", "fixed_mask_n"] + metric_cols)
    writer.writeheader()
    writer.writerows(summary)

print("summary", exp / "summary.csv")
for rec in summary:
    print(rec)
PY

date
echo t4_texture_metrics_complete
