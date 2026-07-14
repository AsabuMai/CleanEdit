#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=04:30:00
#SBATCH -J compabl-sd3
#SBATCH --array=0-2%1
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/final_component_ablation_sd3_%A_%a.log

set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
VARS=(preserve adaptive opsupport)
VARIANT=${VARS[$SLURM_ARRAY_TASK_ID]}

export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$VARIANT" == "opsupport" ]]; then
  export ABLATE=""
  export ADAPT_OVERRIDE_JSON='{"t1_accessory":{"--support-candidate":"attention_x_clean"},"t2_insert":{"--support-candidate":"attention_x_clean"},"t3_text":{"--support-candidate":"attention_x_clean"},"t3_decal":{"--support-candidate":"attention_x_clean"},"t5_material":{"--support-candidate":"attention_x_clean"}}'
else
  export ABLATE=$VARIANT
fi

hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

OUTDIR="$PROJ/outputs/final_component_ablation_non_t4_sd3_${VARIANT}"
case "$OUTDIR" in
  "$PROJ"/outputs/final_component_ablation_non_t4_sd3_*) rm -rf "$OUTDIR" ;;
  *) echo "refusing to clean $OUTDIR" >&2; exit 3 ;;
esac

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_ablation_non_t4_116.json \
OUT=outputs/final_component_ablation_non_t4_sd3_${VARIANT} \
SEED=10 \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
KIND_OVERRIDE_JSON='{"t5_material":{"hedit":"0.82","text":"0.14","rec":"0.62","struct":"0.75"}}' \
SUBJECT_KIND_OVERRIDE_JSON='{"t5_material":{"tar":"7.0","core":"0.85","subj":"0.40","traj":"0.70","max_area":"0.45","min_area":"0.10","relation":"inside_host","layering":"none","keep":"2","dilate":"5"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

echo "[count]"
find "$PROJ/outputs/final_component_ablation_non_t4_sd3_${VARIANT}" -path "*/seed_10/result.png" | wc -l
date
echo final_component_ablation_sd3_${VARIANT}_complete
