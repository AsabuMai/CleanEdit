#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH -J fe135-t3-fairstrong
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_fair_strong_sd3_v1_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub HF_HOME=$PROJ/.cache/huggingface HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date; cd "$PROJ"
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_fair_strong_v1.json OUT=outputs/fe135_t3_fair_strong_sd3_v1 START=0 LIMIT=14 SEED=10 SUBJECT_PRESERVE_MODE=1 SUBJECT_OBJECT_PROVIDER=operation_support_v3 SUBJECT_GROUNDING_METHOD=external_mask T3_REAL_EDIT=0 T3_REF_GUIDANCE=0 T3_SPELL_TEXT=0 T3_COMPACT_PROMPT=0 T3_SURFACE_PROMPT=0 KIND_OVERRIDE_JSON="{\"t3_text\": {\"operation\": \"add_decal\", \"relation\": \"inside_host\", \"layering\": \"none\", \"hedit\": \"1.18\", \"text\": \"0.30\", \"rec\": \"0.36\", \"struct\": \"0.45\", \"core\": \"1.20\", \"subj\": \"0.16\", \"color\": null}, \"t3_decal\": {\"operation\": \"add_decal\", \"relation\": \"on_surface\", \"layering\": \"none\", \"hedit\": \"1.05\", \"text\": \"0.24\", \"rec\": \"0.38\", \"struct\": \"0.45\", \"core\": \"1.12\", \"subj\": \"0.12\", \"color\": null}}" SUBJECT_KIND_OVERRIDE_JSON="{\"t3_text\": {\"tar\": \"8.0\", \"core\": \"1.20\", \"subj\": \"0.16\", \"traj\": \"0.25\", \"max_area\": \"0.42\", \"min_area\": \"0.04\", \"relation\": \"inside_host\", \"layering\": \"none\", \"keep\": \"2\", \"dilate\": \"5\"}, \"t3_decal\": {\"tar\": \"8.0\", \"core\": \"1.12\", \"subj\": \"0.12\", \"traj\": \"0.30\", \"max_area\": \"0.34\", \"min_area\": \"0.03\", \"relation\": \"on_surface\", \"layering\": \"none\", \"keep\": \"2\", \"dilate\": \"5\"}}" "$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
echo "[fair_strong count]"; find "$PROJ/outputs/fe135_t3_fair_strong_sd3_v1" -name result.png | wc -l
date; echo fe135_t3_fair_strong_sd3_v1_complete
