#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH -J attach-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/CleanEdit-source-attachment/logs/source_attachment_sd3_%j.log

set -euo pipefail
R=/cluster/users/grad/2025/25t8103/CleanEdit-source-attachment
A=/cluster/users/grad/2025/25t8103/project
PY=$A/.venv/bin/python
B=$R/.tmp/source_attachment_sd3

cd "$R"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"
test -z "$(git status --porcelain --untracked-files=normal)"
rm -rf "$B"
mkdir -p "$B"

export HF_HOME=$A/.cache/huggingface
export HF_HUB_CACHE=$A/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$A/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$A/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1

CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" run_edit_sd3.py \
  --image "$A/data/paper_images/cat_sitting_in_grass.jpg" \
  --source-prompt "A photo of a cat sitting in grass." \
  --prompt "A photo of the same cat wearing a small golden crown." \
  --output "$B/result.png" --metadata-output "$B/metadata.json" \
  --stats-output "$B/stats.json" --mask-output-dir "$B/masks" \
  --max-image-size 192 --seed 10 --num-inference-steps 3 --n-max 3 \
  --src-guidance-scale 1 --tar-guidance-scale 7.5 \
  --rec-guidance-scale .1 --edit-hedit-guidance-scale .1 \
  --object-mask-provider operation_support_v3 \
  --support-score operation_default --support-temporal-aggregation mean \
  --support-min-area-ratio .01 --support-max-area-ratio .35 \
  --support-keep-components 2 --support-dilate-radius 1 --support-blur-kernel 1 \
  --edit-operation add_object --relation above_host --mask-layering-mode object_contact \
  --final-edit-mask "$A/outputs/pretty_matrix/cat_crown/adaptive_full_generic_support/seed_10/masks/subject_final.png" \
  --final-edit-mask-mode replace --edit-mask-dilate-kernel 3 --edit-mask-erode-kernel 3 \
  --edit-mask-hole-fraction .05 --edit-mask-boundary-noise-scale .1 --edit-mask-smooth-kernel 3 \
  --new-tokens crown --attention-mask-target-words crown --host-tokens cat \
  --source-attachment-release-scale .35

"$PY" - "$B" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text())
stats = json.loads((root / "stats.json").read_text())
assert metadata["source_attachment_release_scale"] == 0.35
assert metadata["shared_core_authoritative"] is True
assert any(row["source_attachment_release_norm"] > 0.0 for row in stats)
assert any(row["source_attachment_release_weight"] == 0.0 for row in stats)
print("SD3 shared source-attachment release active without an extra forward")
PY
