#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:8
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH -J fe135-t3-1024boost
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t3_all_res1024_boost_v1_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub HF_HOME=$PROJ/.cache/huggingface HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date; cd "$PROJ"
MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t3_all42.json \
OUT=outputs/fe135_t3_all_res1024_boost_v1 START=0 LIMIT=42 SEED=10 \
SUBJECT_PRESERVE_MODE=1 SUBJECT_OBJECT_PROVIDER=operation_support_v3 SUBJECT_GROUNDING_METHOD=external_mask \
T3_REAL_EDIT=0 T3_REF_GUIDANCE=0 T3_SPELL_TEXT=0 T3_COMPACT_PROMPT=0 T3_SURFACE_PROMPT=0 \
KIND_OVERRIDE_JSON="{\"t3_text\": {\"operation\": \"add_decal\", \"relation\": \"inside_host\", \"layering\": \"none\", \"hedit\": \"1.12\", \"text\": \"0.26\", \"rec\": \"0.38\", \"struct\": \"0.42\", \"core\": \"1.14\", \"subj\": \"0.20\", \"color\": null}, \"t3_decal\": {\"operation\": \"add_decal\", \"relation\": \"on_surface\", \"layering\": \"none\", \"hedit\": \"0.98\", \"text\": \"0.18\", \"rec\": \"0.38\", \"struct\": \"0.42\", \"core\": \"1.06\", \"subj\": \"0.14\", \"color\": null}}" \
SUBJECT_KIND_OVERRIDE_JSON="{\"t3_text\": {\"tar\": \"8.0\", \"core\": \"1.14\", \"subj\": \"0.20\", \"traj\": \"0.28\", \"max_area\": \"0.36\", \"min_area\": \"0.04\", \"relation\": \"inside_host\", \"layering\": \"none\", \"keep\": \"2\", \"dilate\": \"5\"}, \"t3_decal\": {\"tar\": \"8.0\", \"core\": \"1.06\", \"subj\": \"0.14\", \"traj\": \"0.36\", \"max_area\": \"0.28\", \"min_area\": \"0.03\", \"relation\": \"on_surface\", \"layering\": \"none\", \"keep\": \"2\", \"dilate\": \"5\"}}" \
ADAPT_OVERRIDE_JSON="{\"t3_text\": {\"--max-image-size\": \"1024\"}, \"t3_decal\": {\"--max-image-size\": \"1024\"}}" \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py
echo "[1024_boost count]"; find "$PROJ/outputs/fe135_t3_all_res1024_boost_v1" -name result.png | wc -l
date; echo res1024_boost_complete
