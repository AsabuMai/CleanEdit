#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH -J mask-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/CleanEdit-mask-rules/logs/mask_policy_sd3_%j.log

set -euo pipefail
R=/cluster/users/grad/2025/25t8103/CleanEdit-mask-rules
A=/cluster/users/grad/2025/25t8103/project
PY=$A/.venv/bin/python
B=$R/outputs/diagnostics_20260714/mask_policy_sd3

cd "$R"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"
test -z "$(git status --porcelain --untracked-files=normal)"
test ! -e "$B"
mkdir -p "$B"

export HF_HOME=$A/.cache/huggingface
export HF_HUB_CACHE=$A/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$A/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$A/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1

common=(
  --max-image-size 192 --seed 10 --num-inference-steps 3 --n-max 3
  --src-guidance-scale 1 --tar-guidance-scale 7.5
  --rec-guidance-scale .1 --edit-hedit-guidance-scale .1
  --object-mask-provider semantic --mask-policy operation
  --support-debug-only --photo-prompt-mode off
)

CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" run_edit_sd3.py \
  --image "$A/_baselines/src/FlowEdit/Data/Images/cat_crown.png" \
  --source-prompt 'A gray cat wearing a crown.' \
  --prompt 'A gray cat wearing a black top hat.' \
  --output "$B/replace.png" --metadata-output "$B/replace.json" \
  --stats-output "$B/replace_stats.json" --mask-output-dir "$B/replace_masks" \
  --semantic-base-mask "$R/data/flowedit_compatible_118/sam_support_masks_plural_v2/fe_046_cat_crown_1_black_top_hat_support.png" \
  --edit-operation replace --relation on_head --mask-layering-mode object_contact \
  --new-tokens hat --host-tokens cat --removed-tokens crown "${common[@]}"

CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" run_edit_sd3.py \
  --image "$A/_baselines/src/FlowEdit/Data/Images/corgi.png" \
  --source-prompt 'A brown and white corgi sitting outdoors.' \
  --prompt 'A brown and white corgi made out of lego bricks sitting outdoors.' \
  --output "$B/material.png" --metadata-output "$B/material.json" \
  --stats-output "$B/material_stats.json" --mask-output-dir "$B/material_masks" \
  --semantic-base-mask "$R/data/flowedit_compatible_118/sam_support_masks_plural_v2/fe_080_corgi_1_lego_bricks_support.png" \
  --edit-operation material --relation inside --mask-layering-mode none \
  --new-tokens lego --host-tokens corgi "${common[@]}"

"$PY" - "$B" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
replace = json.loads((root / "replace.json").read_text())
material = json.loads((root / "material.json").read_text())
replace_stats = json.loads((root / "replace_stats.json").read_text())[0]
material_stats = json.loads((root / "material_stats.json").read_text())[0]
assert replace["mask_policy"] == "operation"
assert replace_stats["support_mask_expansion_kernel"] == 1
assert replace_stats["support_removed_required"] == 1
assert replace_stats["support_removed_available"] == 1
assert replace_stats["support_removed_applied"] in {0, 1}
assert material["mask_policy"] == "operation"
assert material_stats["support_mask_policy"] == "appearance_zero_expand"
assert material_stats["support_mask_expansion_kernel"] == 1
print("SD3 replacement/appearance policy smoke passed")
PY
