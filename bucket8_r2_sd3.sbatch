#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J b8r2-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket8_r2_sd3_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"

# Bucket 8 round 2.
# fe_027: raw SAM mask only covered the right butterfly (INST issue inside a
# T4 case) — regenerate with the bucket-1 plural machinery.
KEEP_ALL_BOXES=1 BOX_SCORE_FLOOR=0.28 BOX_SCORE_REL=0.5 \
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket8_027src.json \
OUT_DIR=$PROJ/data/flowedit_compatible_118/sam_support_masks_plural_v2 \
OUT_MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket8_027plural.json \
"$PROJ/.venv/bin/python" scripts/make_flowedit118_sam_support_manifest.py

"$PROJ/.venv/bin/python" - <<PYEOF
import json
new027 = json.load(open("data/flowedit_compatible_135/manifest_bucket8_027plural.json"))
new027 = new027.get("items", new027) if isinstance(new027, dict) else new027
old = json.load(open("data/flowedit_compatible_135/manifest_bucket8_melt.json"))
merged = new027 + [it for it in old if not it["key"].startswith("fe_027")]
json.dump(merged, open("data/flowedit_compatible_135/manifest_bucket8_r2.json","w"), indent=1)
print("merged", [it["key"][:6] for it in merged])
PYEOF

# fe_128/028: the color-similarity gate leaked onto color-similar background
# (gray bird vs gray bokeh) — gate recolor with the tight external SAM mask
# via the existing --edit-color-mask-image path. fe_027 uses its new plural
# mask so both butterflies recolor.
export COLOR_SOURCE_HOST_AWARE=1
export COLOR_SOURCE_FROM_MASK=1
export KEY_OVERRIDE_JSON='{"fe_128":{"--edit-color-mask-image":"data/flowedit_compatible_118/sam_support_masks_all/fe_128_gray_bird_2_red_bird_support.png"},"fe_028":{"--edit-color-mask-image":"data/flowedit_compatible_118/sam_support_masks_all/fe_028_butterfly_1_orange_butterfly_support.png"},"fe_027":{"--edit-color-mask-image":"data/flowedit_compatible_118/sam_support_masks_plural_v2/fe_027_butterflies_1_yellow_support.png"}}'
export ABLATE=no_final_restore
export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket8_r2.json
export OUT=outputs/bucket8_r2_sd3
export SEED=10
export SUBJECT_PRESERVE_MODE=1
export SUBJECT_OBJECT_PROVIDER=operation_support_v3
export SUBJECT_GROUNDING_METHOD=external_mask
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
find "$PROJ/$OUT" -path "*/seed_10/result.png" | wc -l
date
echo bucket8_r2_complete
