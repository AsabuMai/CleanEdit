#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH -J fe118-fxmetrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit118_fixedmask_metrics_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/flowedit118_baselines_fixedmask_metrics_20260619
MASK_DIR=$PROJ/data/flowedit_compatible_118/eval_masks
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi
mkdir -p "$EXP"
python3 prepare_flowedit118_metric_runs.py
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJ/outputs/flowedit118_metric_runs" \
  --csv-output "$EXP/metrics.csv" \
  --json-output "$EXP/metrics.json" \
  --task-names all \
  --method-names all \
  --seeds 10 \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --eval-mask-dir "$MASK_DIR"
.venv/bin/python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

path = Path("experiments/flowedit118_baselines_fixedmask_metrics_20260619/metrics.csv")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    groups[row["method"]].append(row)

def f(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0

cols = [
    "method",
    "n",
    "clip_t",
    "clip_target_minus_source",
    "clip_direction_similarity",
    "local_clip_t_full_prompt",
    "local_clip_target_minus_source",
    "clip_image_source_similarity",
    "dino_source_similarity",
    "bg_dino_source",
    "lpips_full",
    "lpips_outside",
    "bg_lpips",
    "source_l1",
    "inside_l1",
    "outside_l1",
    "bg_l1",
    "source_ssim_luma",
    "bg_ssim_luma",
    "eval_mask_area",
    "bg_mask_area",
]
summary = []
for method, items in sorted(groups.items()):
    rec = {"method": method, "n": len(items)}
    for key in cols[2:]:
        rec[key] = sum(f(row, key) for row in items) / max(1, len(items))
    summary.append(rec)
out = path.with_name("summary_by_method.csv")
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=cols)
    writer.writeheader()
    writer.writerows(summary)
print("summary", out)
for rec in summary:
    print(rec)
PY
date
echo flowedit118_fixedmask_metrics_complete
