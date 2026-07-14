#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH -J component-reg
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/shared_component_regression_%j.log

set -euo pipefail
R=/cluster/users/grad/2025/25t8103/project
B=$R/.tmp/shared_component_20260714
OLD=$R/.tmp/bitwise_cleanup_20260714/out5/post5-add
cd "$R"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"
rm -rf "$B"
mkdir -p "$B"

.venv/bin/python tests/test_dece_core_component_control.py

common=(
  --image "$R/data/paper_images/cat_sitting_in_grass.jpg"
  --source-prompt "A photo of a cat sitting in grass."
  --prompt "A photo of the same cat wearing a small golden crown."
  --max-image-size 192 --seed 10 --num-inference-steps 3 --n-max 3
  --src-guidance-scale 1 --tar-guidance-scale 7.5
  --rec-guidance-scale .1 --edit-hedit-guidance-scale .1
  --object-mask-provider operation_support_v3
  --support-score operation_default --support-temporal-aggregation mean
  --support-min-area-ratio .01 --support-max-area-ratio .35
  --support-keep-components 2 --support-dilate-radius 1 --support-blur-kernel 1
  --edit-operation add_object --relation above_host --mask-layering-mode object_contact
  --final-edit-mask "$R/outputs/pretty_matrix/cat_crown/adaptive_full_generic_support/seed_10/masks/subject_final.png"
  --final-edit-mask-mode replace --edit-mask-dilate-kernel 3 --edit-mask-erode-kernel 3
  --edit-mask-hole-fraction .05 --edit-mask-boundary-noise-scale .1 --edit-mask-smooth-kernel 3
  --new-tokens crown --attention-mask-target-words crown --host-tokens cat
)

run_case() {
  local name=$1
  shift
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python run_edit_sd3.py \
    "${common[@]}" --output "$B/$name.png" --stats-output "$B/$name.stats.json" \
    --metadata-output "$B/$name.metadata.json" --mask-output-dir "$B/$name.masks" "$@"
}

run_case debug --save-support-debug --support-debug-only
cmp "$OLD.png" "$B/debug.png"
cmp "$OLD.stats.json" "$B/debug.stats.json"
diff -qr "$OLD.masks" "$B/debug.masks"

run_case legacy
run_case shadow --shared-core-shadow
run_case authoritative --shared-core-authoritative
cmp "$B/legacy.png" "$B/shadow.png"
cmp "$B/legacy.png" "$B/authoritative.png"

run_case component \
  --shared-core-authoritative --adaptive-clean-control --adaptive-component-control \
  --adaptive-rmsgap-mode normgate --adaptive-rmsgap-dead-zone .1 \
  --adaptive-edit-gain 1 --adaptive-edit-weight-min 1 --adaptive-edit-weight-max 2

.venv/bin/python - "$B" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
shadow = json.loads((root / "shadow.stats.json").read_text())
active = [row for row in shadow if row.get("shared_core_shadow_active")]
assert active and max(row["shared_core_shadow_total_rel"] for row in active) <= 1e-6
component = json.loads((root / "component.stats.json").read_text())
active_component = [row for row in component if row.get("adaptive_component_active_count", 0) > 0]
assert active_component, "component controller never became active"
assert all(row.get("shared_core_authoritative_active") for row in component)
print("default bitwise: PASS")
print("shared authoritative parity: PASS")
print("component controller active: PASS")
PY
