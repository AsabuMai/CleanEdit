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
ATTACH_SCALE=${ATTACH_SCALE:-0.35}
BASE=${ATTACH_OUT:-outputs/diagnostics_20260714/source_attachment_release_s035}

cd "$R"
if [[ "${VERIFY_ONLY:-0}" == "1" ]]; then
    [[ "$(hostname)" == "a100-01.gpu01.cis.k.hosei.ac.jp" || "$(hostname)" == "h100-01.gpu01.cis.k.hosei.ac.jp" ]]
else
    test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"
fi
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
export ADAPT_OVERRIDE_JSON="{\"recolor\":{\"--source-attachment-release-scale\":\"$ATTACH_SCALE\"},\"material\":{\"--source-attachment-release-scale\":\"$ATTACH_SCALE\"}}"

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

if [[ "${VERIFY_ONLY:-0}" != "1" ]]; then
    run_bucket b4_flux_noed data/flowedit_compatible_135/evidence_20260714/manifest_b4_flux_noed.json
    run_bucket b6_flux_bgrep data/flowedit_compatible_135/evidence_20260714/manifest_b6_flux_bgrep.json
    run_bucket b8_melt_scale data/flowedit_compatible_135/evidence_20260714/manifest_b8_melt_scale.json
fi

"$PY" - "$R/$BASE" "$ATTACH_SCALE" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
release_scale = float(sys.argv[2])
metadata = sorted(root.glob("**/metadata.json"))
assert len(metadata) == 18, len(metadata)
expected_active = 0
for path in metadata:
    data = json.loads(path.read_text())
    assert data["seed"] == 10
    assert data["num_inference_steps"] == 12
    assert data["n_max"] == 10
    assert data["final_postprocess_mode"] == "none"
    assert data["postprocess_applied"] is False
    assert data["evaluation_eligible"] is False
    assert data["paper_use"] is False
    operation = data["support_stats"]["support_edit_operation"]
    expected = release_scale if operation in {"recolor", "replace"} else 0.0
    assert data["source_attachment_release_scale"] == expected, path
    expected_active += int(expected > 0.0)
stats = sorted(root.glob("**/stats.json"))
assert len(stats) == 18, len(stats)
active = 0
for path in stats:
    rows = json.loads(path.read_text())
    if any(row.get("source_attachment_release_norm", 0.0) > 0.0 for row in rows):
        active += 1
assert active == expected_active, (active, expected_active)
print(
    f"verified 18 fixed-budget runs at scale={release_scale}; "
    f"source release active on {active} recolor/material runs"
)
PY

if [[ -n "${PAIR_WITH:-}" ]]; then
"$PY" - "$R/$BASE" "$R/$PAIR_WITH" <<'PY'
import json, sys
from pathlib import Path

control_root, release_root = map(Path, sys.argv[1:])

def index(root):
    rows = {}
    for path in root.glob("**/metadata.json"):
        relative = path.relative_to(root)
        rows[tuple(relative.parts[:2])] = path
    return rows

control = index(control_root)
release = index(release_root)
assert len(control) == 18 and control.keys() == release.keys()
active_changed = 0
inactive_identical = 0
for key in sorted(control):
    control_path, release_path = control[key], release[key]
    left = json.loads(control_path.read_text())
    right = json.loads(release_path.read_text())
    for field in (
        "seed", "num_inference_steps", "n_max", "source_image_sha256",
        "source_prompt", "prompt", "final_postprocess_mode", "postprocess_applied",
        "support_control_mode", "support_mode", "support_stats",
    ):
        assert left[field] == right[field], (key, field)
    assert left["source_attachment_release_scale"] == 0.0, key
    release_scale = right["source_attachment_release_scale"]
    for mask_name in ("M_edit.png", "M_core.png", "M_preserve.png"):
        left_mask = control_path.parent / "masks" / mask_name
        right_mask = release_path.parent / "masks" / mask_name
        assert left_mask.read_bytes() == right_mask.read_bytes(), (key, mask_name)
    same_result = (
        (control_path.parent / "result.png").read_bytes()
        == (release_path.parent / "result.png").read_bytes()
    )
    if release_scale > 0.0:
        assert not same_result, key
        active_changed += 1
    else:
        assert same_result, key
        inactive_identical += 1
assert (active_changed, inactive_identical) == (15, 3)
print("paired control verified: 15 active changed, 3 inactive bitwise identical")
PY
fi
