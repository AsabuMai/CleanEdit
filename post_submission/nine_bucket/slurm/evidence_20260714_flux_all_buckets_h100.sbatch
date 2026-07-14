#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -J ev-9bucket-flux
#SBATCH -o /cluster/users/grad/2025/25t8103/CleanEdit-evidence/logs/evidence_9bucket_flux_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/CleanEdit-evidence
ASSET_ROOT=/cluster/users/grad/2025/25t8103/project
PY=$ASSET_ROOT/.venv-h100/bin/python
LOCK=$PROJ/data/flowedit_compatible_135/evidence_20260714/nine_bucket_flux_evidence.lock.json
OUTPUT_ROOT=outputs/evidence_20260714_clean_v2

cd "$PROJ"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"
test -z "$(git status --porcelain --untracked-files=normal)"
COMMIT=$(git rev-parse HEAD)
LOCK_SHA=$(sha256sum "$LOCK" | awk '{print $1}')

export HF_HOME=$ASSET_ROOT/.cache/huggingface
export HF_HUB_CACHE=$ASSET_ROOT/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$ASSET_ROOT/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$ASSET_ROOT/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_bucket() (
    bucket=$1
    recipe=$2
    manifest=$3
    export EVIDENCE_BUCKET=$bucket EVIDENCE_RECIPE=$recipe EVIDENCE_LOCK_SHA256=$LOCK_SHA
    export MANIFEST=$PROJ/$manifest
    export FLUX_OUT=$OUTPUT_ROOT/$bucket
    export ABLATE=no_final_postprocess RUN_PURPOSE=evaluation SEED=10
    export FLUX_LOCAL_FILES_ONLY=1 FLUX_OFFLOAD=model FLUX_STEPS=12 FLUX_N_MAX=10
    export T3_NEG_FROM_REMOVED=0 T3_COMPACT_PROMPT=0 T3_SPELL_TEXT=0 T3_SURFACE_PROMPT=0
    export T5_EXPAND_EDIT_MASK=1 T5_EXPAND_EDIT_RADIUS=21 T5_EXPAND_EDIT_MASK_BLUR=6
    export KIND_OVERRIDE_JSON='' KEY_OVERRIDE_JSON='' MASK_OVERRIDE_JSON=''
    export T5_POSE_PROMPT=1 T5_AUTO_PROMPT_PREFIX=1 T5_PROMPT_PREFIX=''
    export T5_NEGATIVE_PROMPT='fur, animal fur, whiskers, hair strands, fuzzy outline, floating thin lines, text, letters, logo, watermark'

    case "$bucket" in
        b3_flux_ghost)
            export T3_NEG_FROM_REMOVED=1 T3_COMPACT_PROMPT=1 T3_SPELL_TEXT=1
            export MASK_OVERRIDE_JSON='{"fe_119":"data/flowedit_compatible_135/repair_allpass_v1/masks/fe_119_gas_station_2_iccv_C5_exact_text_needs_renderer_support.png","fe_134":"data/flowedit_compatible_135/repair_allpass_v1/masks/fe_134_groceries_4_eccv_C5_exact_text_needs_renderer_support.png","fe_238":"data/flowedit_compatible_135/repair_allpass_v1/masks/fe_238_stop_arrow_1_cvpr_C6_short_text_needs_text_aware_refine_support.png"}'
            ;;
        b4_flux_noed)
            export KIND_OVERRIDE_JSON='{"t2_insert":{"edit_hedit":1.20,"local_target":1.10},"t4_recolor":{"edit_hedit":0.80,"local_target":0.35}}'
            ;;
        b6_flux_bgrep|b7_half|b8_melt_scale)
            export T5_EXPAND_EDIT_MASK=0
            export KIND_OVERRIDE_JSON='{"t5_material":{"outside_lock":0.45}}'
            ;;
    esac
    if test "$bucket" = b6_flux_bgrep; then
        export KEY_OVERRIDE_JSON='{"fe_130":{"--edit-hedit-guidance-scale":"1.55"}}'
    fi
    "$PY" scripts/pie_batch_flux_kindaware_v11.py
)

run_bucket b1_inst instance_final_v3 data/flowedit_compatible_135/evidence_20260714/manifest_b1_inst.json
run_bucket b2_sd3_text_control flux_canonical_text_control data/flowedit_compatible_135/evidence_20260714/manifest_b2_sd3_text_control.json
run_bucket b3_flux_ghost best_single_pass_text_nofinal data/flowedit_compatible_135/evidence_20260714/manifest_b3_flux_ghost.json
run_bucket b4_flux_noed kind_budget_final data/flowedit_compatible_135/evidence_20260714/manifest_b4_flux_noed.json
run_bucket b5_sd3_leak_control flux_canonical_recolor_control data/flowedit_compatible_135/evidence_20260714/manifest_b5_sd3_leak_control.json
run_bucket b6_flux_bgrep zero_expand_outside_lock_final data/flowedit_compatible_135/evidence_20260714/manifest_b6_flux_bgrep.json
run_bucket b7_half full_body_mask_zero_expand data/flowedit_compatible_135/evidence_20260714/manifest_b7_half.json
run_bucket b8_melt_scale flux_support_control_zero_expand data/flowedit_compatible_135/evidence_20260714/manifest_b8_melt_scale.json
run_bucket b9_resid removed_token_support data/flowedit_compatible_135/evidence_20260714/manifest_b9_resid.json

"$PY" scripts/verify_nine_bucket_flux_evidence.py \
    --lock "$LOCK" \
    --output-root "$PROJ/$OUTPUT_ROOT" \
    --expected-commit "$COMMIT" \
    --report "$PROJ/$OUTPUT_ROOT/verification_report.json"
date
echo evidence_9bucket_flux_complete
