#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH -J fe135-ablmet24
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_ablation_metrics_20260624_%j.log
set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_sweep_subset.json
RUNS=$PROJ/outputs/flowedit135_ablation_metric_runs_20260624
EXP=$PROJ/experiments/flowedit135_ablation_metrics_20260624
MASK_DIR=$PROJ/data/flowedit_compatible_135/eval_masks
LOCAL_PROMPTS=$PROJ/data/flowedit_compatible_135/local_target_prompts.json
METHODS="ours_sd3 ours_flux sd3abl_adaptive sd3abl_opsupport sd3abl_nofinal fluxabl_adaptive fluxabl_opsupport fluxabl_nofinal"

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
"$PROJ/.venv/bin/python" prepare_flowedit135_metric_runs.py \
  --manifest "$MANIFEST" \
  --out-dir "$RUNS" \
  --methods "$METHODS"

"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$RUNS" \
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
import csv, json
from collections import defaultdict
from pathlib import Path

path = Path("experiments/flowedit135_ablation_metrics_20260624/metrics.csv")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    groups[row["method"]].append(row)

cols = [
    "method", "method_display", "n", "complete_n", "fixed_mask_n", "local_clip_t_n",
    "clip_t", "clip_target_minus_source", "clip_direction_similarity", "local_clip_t",
    "local_clip_t_full_prompt", "local_clip_target_minus_source", "clip_image_source_similarity",
    "dino_source_similarity", "bg_dino_source", "lpips_full", "lpips_outside", "bg_lpips",
    "source_l1", "inside_l1", "outside_l1", "bg_l1", "source_ssim_luma", "bg_ssim_luma",
    "eval_mask_area", "dilated_edit_mask_area", "bg_mask_area",
]
metric_cols = cols[6:]

def mean(items, key):
    vals = []
    for row in items:
        value = row.get(key, "")
        if value == "":
            continue
        vals.append(float(value))
    return "" if not vals else sum(vals) / len(vals)

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
        rec[key] = mean(items, key)
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
        rec[key] = mean(items, key)
    family_summary.append(rec)

family_out = path.with_name("summary_by_family_method.csv")
with family_out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=family_cols)
    writer.writeheader()
    writer.writerows(family_summary)

payload = {
    "rows": len(rows),
    "methods": sorted(groups),
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
print("rows", len(rows))
print("summary", out)
print("family_summary", family_out)
for rec in summary:
    print(rec)
expected = 42
bad = [
    rec for rec in summary
    if rec["n"] != expected or rec["complete_n"] != expected
    or rec["fixed_mask_n"] != expected or rec["local_clip_t_n"] != expected
]
if bad:
    print("bad", bad)
    raise SystemExit(3)
PY

date
echo fe135_ablation_metrics_20260624_complete
