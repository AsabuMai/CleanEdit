#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH -J fe135-t3-ccfix
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_cc_fix_lowstruct_v1_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub HF_HOME=$PROJ/.cache/huggingface HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date; cd "$PROJ"
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_cc_fix.json OUT=outputs/fe135_t3_cc_fix_lowstruct_v1 START=0 LIMIT=6 SEED=10 SUBJECT_PRESERVE_MODE=1 SUBJECT_OBJECT_PROVIDER=operation_support_v3 SUBJECT_GROUNDING_METHOD=external_mask T3_REAL_EDIT=0 T3_REF_GUIDANCE=0 T3_SPELL_TEXT=0 T3_COMPACT_PROMPT=0 T3_SURFACE_PROMPT=0 KIND_OVERRIDE_JSON="{\"t3_text\": {\"operation\": \"add_decal\", \"relation\": \"inside_host\", \"layering\": \"none\", \"hedit\": \"1.05\", \"text\": \"0.18\", \"rec\": \"0.25\", \"struct\": \"0.20\", \"core\": \"1.05\", \"subj\": \"0.30\", \"color\": null}, \"t3_decal\": {\"operation\": \"add_decal\", \"relation\": \"on_surface\", \"layering\": \"none\", \"hedit\": \"0.85\", \"text\": \"0.10\", \"rec\": \"0.25\", \"struct\": \"0.20\", \"core\": \"0.95\", \"subj\": \"0.20\", \"color\": null}}" "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
echo "[cc_fix count]"; find "$PROJ/outputs/fe135_t3_cc_fix_lowstruct_v1" -name result.png | wc -l
date; echo cc_fix_complete
