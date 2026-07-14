#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=01:10:00
#SBATCH -J mask-policy
#SBATCH -o /cluster/users/grad/2025/25t8103/CleanEdit-mask-rules/logs/mask_policy_flux_%j.log

set -euo pipefail
R=/cluster/users/grad/2025/25t8103/CleanEdit-mask-rules
A=/cluster/users/grad/2025/25t8103/project
PY=$A/.venv-h100/bin/python
BASE=$R/outputs/diagnostics_20260714/mask_policy_flux
MANIFEST=$BASE/manifest.json

cd "$R"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"
test -z "$(git status --porcelain --untracked-files=normal)"
test ! -e "$BASE"
mkdir -p "$BASE"

"$PY" - "$R" "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path

root, output = map(Path, sys.argv[1:])
selected = {
    "b1_inst": {"fe_195_parrots2_1_crown", "fe_197_penguins_2_sculpture"},
    "b4_flux_noed": {
        "fe_010_beer_glass_1_cocktail",
        "fe_019_brown_owl_1_white_owl",
        "fe_180_milk_4_whipped_cream",
        "fe_207_pizza_tomato_olive_1_pepperoni",
    },
    "b7_half": {"fe_080_corgi_1_lego_bricks", "fe_081_corgi_2_wooden_sculpture"},
    "b8_melt_scale": {"fe_027_butterflies_1_yellow", "fe_130_gray_bird_4_golden_sculpture"},
    "b9_resid": {"fe_046_cat_crown_1_black_top_hat"},
}
rows = []
for bucket, keys in selected.items():
    path = root / "data/flowedit_compatible_135/evidence_20260714" / f"manifest_{bucket}.json"
    rows.extend(row for row in json.loads(path.read_text()) if row["key"] in keys)
assert len(rows) == 11, len(rows)
output.write_text(json.dumps(rows, indent=2) + "\n")
PY

export HF_HOME=$A/.cache/huggingface
export HF_HUB_CACHE=$A/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$A/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$A/.cache/huggingface/hub
export FLUX_CACHE_DIR=$A/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLEANEDIT_ROOT=$R EXPECTED_PROJECT_ROOT=$R MANIFEST
export ABLATE=no_final_postprocess RUN_PURPOSE=diagnostic SEED=10
export FLUX_LOCAL_FILES_ONLY=1 FLUX_OFFLOAD=model FLUX_STEPS=12 FLUX_N_MAX=10
export T5_EXPAND_EDIT_MASK=1 T5_EXPAND_EDIT_RADIUS=21 T5_EXPAND_EDIT_MASK_BLUR=6
export T5_NEGATIVE_PROMPT='fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark'

for policy in legacy operation; do
    export MASK_POLICY=$policy
    export FLUX_OUT=$BASE/$policy
    "$PY" scripts/pie_batch_flux_kindaware_v11.py
done

"$PY" - "$BASE" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = sorted(root.glob("legacy/**/metadata.json")) + sorted(root.glob("operation/**/metadata.json"))
assert len(metadata) == 22, len(metadata)
for path in metadata:
    data = json.loads(path.read_text())
    assert data["seed"] == 10
    assert data["num_inference_steps"] == 12
    assert data["n_max"] == 10
    assert data["final_postprocess_mode"] == "none"
    assert data["postprocess_applied"] is False
    assert data["evaluation_eligible"] is False
    assert data["attention_support_forward_count"] == 2
    support = data["support_stats"]
    policy = path.relative_to(root).parts[0]
    assert support["support_mask_policy_mode"] == policy
    if policy == "operation":
        operation = data["edit_operation"]
        expected_kernel = 5 if operation in {"add_object", "add_decal", "shape_change"} else 1
        assert support["support_mask_expansion_kernel"] == expected_kernel, path
        if operation == "replace":
            assert support["support_removed_required"] == 1
            assert data["removed_tokens"]
print("verified 11 paired cases, equal 12/10 budget, no final postprocess")
PY
