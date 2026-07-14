#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH -J fe135-fxmet23
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit135_fixedmask_metrics_%j.log
set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/flowedit135_fixedmask_metrics_20260623
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
mask_count=$(find "$MASK_DIR" -maxdepth 1 -name '*_eval_mask.png' 2>/dev/null | wc -l || true)
if [[ "$mask_count" != "135" ]]; then
  echo "building FE135 fixed eval masks: current=$mask_count"
  "$PROJ/.venv/bin/python" make_flowedit135_fixed_eval_masks.py
fi

python3 prepare_flowedit135_metric_runs.py
"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJ/outputs/flowedit135_metric_runs" \
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

"$PROJ/.venv/bin/python" - <<'PY'
import csv
import json
from collections import defaultdict
from pathlib import Path

path = Path("experiments/flowedit135_fixedmask_metrics_20260623/metrics.csv")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    groups[row["method"]].append(row)

def values(items, key):
    out = []
    for row in items:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            out.append(float(value))
        except ValueError:
            pass
    return out

def mean_or_blank(items, key):
    vals = values(items, key)
    return "" if not vals else sum(vals) / len(vals)

cols = [
    "method",
    "method_display",
    "n",
    "complete_n",
    "fixed_mask_n",
    "local_clip_t_n",
    "clip_t",
    "clip_target_minus_source",
    "clip_direction_similarity",
    "local_clip_t",
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
    "dilated_edit_mask_area",
    "bg_mask_area",
]
metric_cols = cols[6:]
summary = []
for method, items in sorted(groups.items()):
    rec = {
        "method": method,
        "method_display": items[0].get("method_display", method),
        "n": len(items),
        "complete_n": sum(str(row.get("complete")) == "True" for row in items),
        "fixed_mask_n": sum(row.get("mask_source") == "fixed_eval_mask" for row in items),
        "local_clip_t_n": sum(1 for row in items if row.get("local_clip_t")),
    }
    for key in metric_cols:
        rec[key] = mean_or_blank(items, key)
    summary.append(rec)

out = path.with_name("summary_by_method.csv")
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=cols)
    writer.writeheader()
    writer.writerows(summary)

by_family = defaultdict(list)
for row in rows:
    by_family[(row.get("family_label", ""), row.get("method", ""))].append(row)
family_cols = ["family_label"] + cols
family_summary = []
for (family, method), items in sorted(by_family.items()):
    rec = {
        "family_label": family,
        "method": method,
        "method_display": items[0].get("method_display", method),
        "n": len(items),
        "complete_n": sum(str(row.get("complete")) == "True" for row in items),
        "fixed_mask_n": sum(row.get("mask_source") == "fixed_eval_mask" for row in items),
        "local_clip_t_n": sum(1 for row in items if row.get("local_clip_t")),
    }
    for key in metric_cols:
        rec[key] = mean_or_blank(items, key)
    family_summary.append(rec)

family_out = path.with_name("summary_by_family_method.csv")
with family_out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=family_cols)
    writer.writeheader()
    writer.writerows(family_summary)

payload = {
    "rows": len(rows),
    "methods": sorted(groups),
    "summary": summary,
    "missing_local_clip_t": [
        {"method": row.get("method"), "task": row.get("task")}
        for row in rows
        if not row.get("local_clip_t")
    ],
    "missing_fixed_mask": [
        {"method": row.get("method"), "task": row.get("task"), "mask_source": row.get("mask_source")}
        for row in rows
        if row.get("mask_source") != "fixed_eval_mask"
    ],
}
(path.parent / "metric_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("summary", out)
print("family_summary", family_out)
for rec in summary:
    print(rec)
expected = 135
bad_n = [rec for rec in summary if rec["n"] != expected or rec["complete_n"] != expected]
bad_masks = [rec for rec in summary if rec["fixed_mask_n"] != expected]
bad_local = [rec for rec in summary if rec["local_clip_t_n"] != expected]
if bad_n or bad_masks or bad_local:
    print("bad_n", bad_n)
    print("bad_masks", bad_masks)
    print("bad_local", bad_local)
    raise SystemExit(3)
PY

date
echo flowedit135_fixedmask_metrics_complete
