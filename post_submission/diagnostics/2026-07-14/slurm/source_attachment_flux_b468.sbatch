#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH -J attach-b468
#SBATCH -o /cluster/users/grad/2025/25t8103/CleanEdit-source-attachment/logs/source_attachment_flux_b468_%j.log

set -euo pipefail
R=/cluster/users/grad/2025/25t8103/CleanEdit-source-attachment
A=/cluster/users/grad/2025/25t8103/project
PY=$A/.venv-h100/bin/python
BASE=outputs/diagnostics_20260714/source_attachment_release_s035

cd "$R"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"
test -z "$(git status --porcelain --untracked-files=normal)"

export HF_HOME=$A/.cache/huggingface
export HF_HUB_CACHE=$A/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$A/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$A/.cache/huggingface/hub
export FLUX_CACHE_DIR=$A/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLEANEDIT_ROOT=$R EXPECTED_PROJECT_ROOT=$R
export ABLATE=no_final_postprocess RUN_PURPOSE=diagnostic SEED=10
export FLUX_LOCAL_FILES_ONLY=1 FLUX_OFFLOAD=model FLUX_STEPS=12 FLUX_N_MAX=10
export T3_NEG_FROM_REMOVED=0 T3_COMPACT_PROMPT=0 T3_SPELL_TEXT=0 T3_SURFACE_PROMPT=0
export T5_POSE_PROMPT=1 T5_AUTO_PROMPT_PREFIX=1 T5_PROMPT_PREFIX=''
export T5_EXPAND_EDIT_RADIUS=21 T5_EXPAND_EDIT_MASK_BLUR=6
export MASK_OVERRIDE_JSON=''
export ADAPT_OVERRIDE_JSON='{"recolor":{"--source-attachment-release-scale":"0.35"},"material":{"--source-attachment-release-scale":"0.35"}}'

run_bucket() (
    bucket=$1
    manifest=$2
    export MANIFEST=$R/$manifest
    export FLUX_OUT=$BASE/$bucket
    export T5_EXPAND_EDIT_MASK=1
    export T5_NEGATIVE_PROMPT='fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark'
    export KIND_OVERRIDE_JSON='' KEY_OVERRIDE_JSON=''
    case "$bucket" in
        b4_flux_noed)
            export KIND_OVERRIDE_JSON='{"t2_insert":{"edit_hedit":1.20,"local_target":1.10},"t4_recolor":{"edit_hedit":0.80,"local_target":0.35}}'
            ;;
        b6_flux_bgrep)
            export T5_EXPAND_EDIT_MASK=0
            export T5_NEGATIVE_PROMPT='fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark, duplicated figure, second person, oversized figure'
            export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45}}'
            export KEY_OVERRIDE_JSON='{"fe_130":{"--edit-hedit-guidance-scale":"1.55"}}'
            ;;
        b8_melt_scale)
            export T5_EXPAND_EDIT_MASK=0
            export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45}}'
            ;;
    esac
    "$PY" scripts/pie_batch_flux_kindaware_v11.py
)

run_bucket b4_flux_noed data/flowedit_compatible_135/evidence_20260714/manifest_b4_flux_noed.json
run_bucket b6_flux_bgrep data/flowedit_compatible_135/evidence_20260714/manifest_b6_flux_bgrep.json
run_bucket b8_melt_scale data/flowedit_compatible_135/evidence_20260714/manifest_b8_melt_scale.json

"$PY" - "$R/$BASE" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = sorted(root.glob("**/metadata.json"))
assert len(metadata) == 18, len(metadata)
for path in metadata:
    data = json.loads(path.read_text())
    assert data["seed"] == 10
    assert data["num_inference_steps"] == 12
    assert data["n_max"] == 10
    assert data["final_postprocess_mode"] == "none"
    assert data["postprocess_applied"] is False
    assert data["evaluation_eligible"] is False
    assert data["paper_use"] is False
    operation = data["edit_operation"]
    expected = 0.35 if operation in {"recolor", "replace"} else 0.0
    assert data["source_attachment_release_scale"] == expected, path
stats = sorted(root.glob("**/stats.json"))
assert len(stats) == 18, len(stats)
active = 0
for path in stats:
    rows = json.loads(path.read_text())
    if any(row.get("source_attachment_release_norm", 0.0) > 0.0 for row in rows):
        active += 1
assert active == 16, active
print("verified 18 fixed-budget runs; source release active on 16 recolor/material runs")
PY
