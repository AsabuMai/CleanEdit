#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J component-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/shared_component_flux_%j.log

set -euo pipefail
R=/cluster/users/grad/2025/25t8103/project
PY=$R/.venv-h100/bin/python
BASE=outputs/diagnostics_20260714/shared_component_flux
cd "$R"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"

export HF_HOME=$R/.cache/huggingface
export HF_HUB_CACHE=$R/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$R/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$R/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABLATE=no_final_postprocess RUN_PURPOSE=diagnostic
export SEED=10 FLUX_LOCAL_FILES_ONLY=1 FLUX_OFFLOAD=model FLUX_STEPS=12 FLUX_N_MAX=10
export ADAPT_OVERRIDE_JSON='{"add":{"--adaptive-rmsgap-mode":"normgate","--adaptive-rmsgap-dead-zone":"0.1"},"material":{"--adaptive-rmsgap-mode":"normgate","--adaptive-rmsgap-dead-zone":"0.1"}}'

run_pair() {
  local name=$1
  export ADAPTIVE_COMPONENT_CONTROL=0 FLUX_OUT=$BASE/${name}_global
  "$PY" scripts/pie_batch_flux_kindaware_v11.py
  export ADAPTIVE_COMPONENT_CONTROL=1 FLUX_OUT=$BASE/${name}_component
  "$PY" scripts/pie_batch_flux_kindaware_v11.py
}

export MANIFEST=$R/data/flowedit_compatible_135/manifest_bucket1_inst_v2.json
export START=5 LIMIT=1
export T5_EXPAND_EDIT_MASK=1 T5_EXPAND_EDIT_RADIUS=21 T5_EXPAND_EDIT_MASK_BLUR=6
export T5_POSE_PROMPT=1 T5_AUTO_PROMPT_PREFIX=1 T5_PROMPT_PREFIX=""
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark"
unset KIND_OVERRIDE_JSON KEY_OVERRIDE_JSON
run_pair fe195

export MANIFEST=$R/data/flowedit_compatible_135/manifest_bucket6_r3.json
export START=0 LIMIT=3 T5_EXPAND_EDIT_MASK=0
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark, duplicated figure, second person, oversized figure"
export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45}}'
export KEY_OVERRIDE_JSON='{"fe_130":{"--edit-hedit-guidance-scale":"1.55"}}'
run_pair bucket6_hard3

export MANIFEST=$R/data/flowedit_compatible_135/manifest_bucket9_resid.json
export START=0 LIMIT=1 T5_EXPAND_EDIT_MASK=1
export T5_NEGATIVE_PROMPT="fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark"
unset KIND_OVERRIDE_JSON KEY_OVERRIDE_JSON
run_pair fe046

"$PY" - "$R/$BASE" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = sorted(root.glob("**/metadata.json"))
assert len(metadata) == 10, len(metadata)
for path in metadata:
    data = json.loads(path.read_text())
    assert data["final_postprocess_mode"] == "none"
    assert data["postprocess_applied"] is False
    assert data["evaluation_eligible"] is False
    assert data["paper_use"] is False
component_stats = sorted(root.glob("*_component/**/stats.json"))
assert len(component_stats) == 5, len(component_stats)
for path in component_stats:
    rows = json.loads(path.read_text())
    assert any(row.get("adaptive_component_count", 0) > 0 for row in rows), path
print("verified 5 paired global/component diagnostic cases")
PY
