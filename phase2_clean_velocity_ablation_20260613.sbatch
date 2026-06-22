#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH -J p2-clean-vel-ablate
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_clean_velocity_ablation_%j.log

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
RUN_TAG="phase2_clean_velocity_ablation_20260613"
RUN_ID="${RUN_TAG}_${SLURM_JOB_ID:-manual}"
RUN_DIR="$EXP/$RUN_ID"
MANIFEST="$RUN_DIR/command_manifest.txt"
SUMMARY="$RUN_DIR/sd3_batch_summary.json"
METRICS_CSV="$RUN_DIR/clean_velocity_ablation_metrics.csv"
METRICS_JSON="$RUN_DIR/clean_velocity_ablation_metrics.json"
SUMMARY_MD="$RUN_DIR/clean_velocity_ablation_summary.md"

TASKS="${TASKS:-cat_crown bowl_apple_inside mug_heart red_office_chair_to_blue_office_chair pillow_same_color_cable_knit}"
SEEDS="${SEEDS:-10}"
BASE_METHOD="support_v3_controller_rmsgap"
ABLATION_METHODS="
support_v3_controller_rmsgap
support_v3_controller_rmsgap_ablate_rf_diff_editfield
support_v3_controller_rmsgap_ablate_legacy_conversion
support_v3_controller_rmsgap_ablate_support_attn_clean
support_v3_controller_rmsgap_ablate_support_attn_velocity
"

export PYTHON="$PROJECT/.venv/bin/python"
export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_TELEMETRY=1
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
export N_MAX="${N_MAX:-24}"
export MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-512}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export REGENERATE_MASKS="${REGENERATE_MASKS:-0}"
export REUSE_SEMANTIC_MASKS="${REUSE_SEMANTIC_MASKS:-1}"

cd "$PROJECT"
mkdir -p "$RUN_DIR" "$PROJECT/logs"
rm -f "$MANIFEST"

host="$(hostname)"
echo "host=$host"
date
echo "run_dir=$RUN_DIR"
echo "tasks=$TASKS"
echo "seeds=$SEEDS"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi
nvidia-smi || true

prepare_variant() {
  local suffix="$1"
  local support_candidate="$2"
  local edit_field_mode="$3"
  local velocity_conversion="$4"
  local variant_manifest="$RUN_DIR/${suffix#_}_manifest.txt"

  rm -f "$variant_manifest"
  echo "== prepare ${suffix} support=${support_candidate} edit_field=${edit_field_mode} velocity_conversion=${velocity_conversion} =="
  METHOD_NAME_SUFFIX="$suffix" \
  SUPPORT_V3_CANDIDATE_OVERRIDE="$support_candidate" \
  BATCH_MANIFEST="$variant_manifest" \
  PREPARE_ONLY=1 \
  TASKS="$TASKS" \
  METHODS="$BASE_METHOD" \
  SEEDS="$SEEDS" \
  bash scripts/run_pretty_matrix.sh

  "$PYTHON" - "$variant_manifest" "$edit_field_mode" "$velocity_conversion" <<'PY'
from pathlib import Path
import shlex
import sys

manifest = Path(sys.argv[1])
edit_field_mode = sys.argv[2]
velocity_conversion = sys.argv[3]

for raw in manifest.read_text(encoding="utf-8").splitlines():
    path = Path(raw.strip())
    if not path.exists():
        raise SystemExit(f"missing command file: {path}")
    command = path.read_text(encoding="utf-8").strip()
    if edit_field_mode != "surrogate" and "--edit-field-mode" not in command:
        command += " --edit-field-mode " + shlex.quote(edit_field_mode)
    if velocity_conversion != "linear_path":
        command = command.replace(
            "--velocity-conversion-mode linear_path",
            "--velocity-conversion-mode " + shlex.quote(velocity_conversion),
        )
    path.write_text(command + "\n", encoding="utf-8")
PY

  cat "$variant_manifest" >> "$MANIFEST"
}

prepare_variant "_ablate_rf_diff_editfield" "operation_default" "rf_diff" "linear_path"
prepare_variant "_ablate_legacy_conversion" "operation_default" "surrogate" "legacy"
prepare_variant "_ablate_support_attn_clean" "attention_x_clean" "surrogate" "linear_path"
prepare_variant "_ablate_support_attn_velocity" "attention_x_velocity" "surrogate" "linear_path"

echo "== command count =="
wc -l "$MANIFEST"

echo "== run ablation batch =="
"$PYTHON" scripts/run_sd3_batch.py \
  --manifest "$MANIFEST" \
  --summary-output "$SUMMARY" \
  --skip-existing

echo "== evaluate ablation metrics =="
"$PYTHON" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$METRICS_CSV" \
  --json-output "$METRICS_JSON" \
  --task-names "$TASKS" \
  --method-names "$ABLATION_METHODS" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --compute-lpips \
  --bg-dilate-px 31

echo "== summarize =="
"$PYTHON" - "$METRICS_CSV" "$SUMMARY_MD" <<'PY'
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys

metrics_csv = Path(sys.argv[1])
summary_md = Path(sys.argv[2])
rows = list(csv.DictReader(metrics_csv.open(newline="", encoding="utf-8")))
metrics = [
    "outside_l1",
    "bg_psnr",
    "bg_lpips_x100",
    "bg_ssim_luma",
    "dino_source_similarity",
    "clip_t",
    "local_clip_t",
    "edit_score",
]

def fnum(value: str) -> float | None:
    if value in {"", "nan", "None", None}:
        return None
    try:
        return float(value)
    except ValueError:
        return None

by_method = defaultdict(list)
for row in rows:
    by_method[row.get("method", "")].append(row)

lines = [
    "# Phase2 Clean/Velocity Ablation Summary",
    "",
    "Scope: smoke ablation over the selected TASKS and SEEDS from the sbatch environment.",
    "",
    "| method | n | outside_l1 | bg_psnr | bg_lpips_x100 | bg_ssim_luma | dino_source | clip_t | local_clip_t | edit_score |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for method in sorted(by_method):
    method_rows = by_method[method]
    vals = {}
    for metric in metrics:
        clean = [v for row in method_rows if (v := fnum(row.get(metric, ""))) is not None]
        vals[metric] = sum(clean) / len(clean) if clean else None
    lines.append(
        "| "
        + " | ".join(
            [
                method,
                str(len(method_rows)),
                *(("" if vals[m] is None else f"{vals[m]:.4f}") for m in metrics),
            ]
        )
        + " |"
    )

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

date
echo phase2_clean_velocity_ablation_complete
