#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH -J b3-2stage
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/bucket3_two_stage_flux_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
cd "$PROJ"
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp"

export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_bucket23_t3.json
export FLUX_OUT=${FLUX_OUT:-outputs/bucket3_two_stage_flux_v1}
export START=${START:-0}
export LIMIT=${LIMIT:-1}
export SEED=${SEED:-10}
export FLUX_LOCAL_FILES_ONLY=1 FLUX_OFFLOAD=model FLUX_STEPS=12 FLUX_N_MAX=10
export T3_TWO_STAGE=1 T3_NEG_FROM_REMOVED=0
export T3_COMPACT_PROMPT=1 T3_SPELL_TEXT=1 T3_SURFACE_PROMPT=0
export T3_ERASE_FILL_SCALE=${T3_ERASE_FILL_SCALE:-1.0}
export T3_ERASE_SUPPRESSION_SCALE=${T3_ERASE_SUPPRESSION_SCALE:-0.55}
export T3_ERASE_RING_REC_SCALE=${T3_ERASE_RING_REC_SCALE:-0.35}
export MASK_OVERRIDE_JSON='{"fe_119":"data/flowedit_compatible_135/repair_allpass_v1/masks/fe_119_gas_station_2_iccv_C5_exact_text_needs_renderer_support.png","fe_134":"data/flowedit_compatible_135/repair_allpass_v1/masks/fe_134_groceries_4_eccv_C5_exact_text_needs_renderer_support.png","fe_238":"data/flowedit_compatible_135/repair_allpass_v1/masks/fe_238_stop_arrow_1_cvpr_C6_short_text_needs_text_aware_refine_support.png"}'

"$PROJ/.venv-h100/bin/python" scripts/pie_batch_flux_kindaware_v11.py
find "$PROJ/$FLUX_OUT" -path '*/dece_rf_flux_two_stage/*/result.png' -print
