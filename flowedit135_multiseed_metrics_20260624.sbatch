#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH -J fe135-msmet
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flowedit135_multiseed_metrics_%j.log
set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/flowedit135_multiseed_metrics_20260624
RUNS=$PROJ/outputs/flowedit135_multiseed_metric_runs
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

"$PROJ/.venv/bin/python" prepare_flowedit135_multiseed_metric_runs.py \
  --out-dir "$RUNS" \
  --methods ours_sd3,ours_flux \
  --seeds 10,11,12

"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$RUNS" \
  --csv-output "$EXP/metrics.csv" \
  --json-output "$EXP/metrics.json" \
  --task-names all \
  --method-names "ours_sd3 ours_flux" \
  --seeds "10 11 12" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31 \
  --eval-mask-dir "$MASK_DIR"

"$PROJ/.venv/bin/python" - <<'PY'
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

path = Path("experiments/flowedit135_multiseed_metrics_20260624/metrics.csv")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
metric_cols = [
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

def mean(xs):
    return "" if not xs else sum(xs) / len(xs)

def std(xs):
    if len(xs) <= 1:
        return ""
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def write_csv(out, fieldnames, records):
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

by_method_seed = defaultdict(list)
by_method = defaultdict(list)
for row in rows:
    by_method_seed[(row["method"], row["seed"])].append(row)
    by_method[row["method"]].append(row)

seed_fields = ["method", "method_display", "seed", "n", "complete_n", "fixed_mask_n", "local_clip_t_n"] + metric_cols
seed_summary = []
for (method, seed), items in sorted(by_method_seed.items()):
    rec = {
        "method": method,
        "method_display": items[0].get("method_display", method),
        "seed": seed,
        "n": len(items),
        "complete_n": sum(str(row.get("complete")) == "True" for row in items),
        "fixed_mask_n": sum(row.get("mask_source") == "fixed_eval_mask" for row in items),
        "local_clip_t_n": sum(1 for row in items if row.get("local_clip_t")),
    }
    for key in metric_cols:
        rec[key] = mean(vals(items, key))
    seed_summary.append(rec)

method_fields = ["method", "method_display", "n", "complete_n", "fixed_mask_n", "local_clip_t_n"]
for key in metric_cols:
    method_fields += [f"{key}_mean", f"{key}_std"]
method_summary = []
for method, items in sorted(by_method.items()):
    rec = {
        "method": method,
        "method_display": items[0].get("method_display", method),
        "n": len(items),
        "complete_n": sum(str(row.get("complete")) == "True" for row in items),
        "fixed_mask_n": sum(row.get("mask_source") == "fixed_eval_mask" for row in items),
        "local_clip_t_n": sum(1 for row in items if row.get("local_clip_t")),
    }
    for key in metric_cols:
        xs = vals(items, key)
        rec[f"{key}_mean"] = mean(xs)
        rec[f"{key}_std"] = std(xs)
    method_summary.append(rec)

out_dir = path.parent
write_csv(out_dir / "summary_by_method_seed.csv", seed_fields, seed_summary)
write_csv(out_dir / "summary_by_method_mean_std.csv", method_fields, method_summary)

audit = {
    "rows": len(rows),
    "methods": sorted(by_method),
    "method_seed_groups": {f"{m}/seed_{s}": len(v) for (m, s), v in sorted(by_method_seed.items())},
    "bad_method_seed_groups": [
        {"method": m, "seed": s, "n": len(v)}
        for (m, s), v in sorted(by_method_seed.items())
        if len(v) != 135
    ],
}
(out_dir / "metric_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print("summary_by_method_seed", out_dir / "summary_by_method_seed.csv")
print("summary_by_method_mean_std", out_dir / "summary_by_method_mean_std.csv")
print(json.dumps(audit, indent=2))
if len(rows) != 810 or audit["bad_method_seed_groups"]:
    raise SystemExit(3)
PY

date
echo flowedit135_multiseed_metrics_complete
