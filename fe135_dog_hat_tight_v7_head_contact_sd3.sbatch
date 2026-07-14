#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J fe135-doghat-v7
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_dog_hat_tight_v7_head_contact_sd3_%j.log

set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date
cd "$PROJ"

EDIT_MASK=data/flowedit_compatible_135/derived_masks/fe_094_dog_6_red_top_hat_v7_head_contact_edit.png

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_dog_hat_tightmask_v7_head_contact.json \
OUT=outputs/fe135_dog_hat_tight_v7_head_contact_sd3 \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
KIND_OVERRIDE_JSON='{"t1_accessory":{"hedit":"1.16","text":"0.14","rec":"0.20","struct":"0.40","core":"1.00","subj":"0.05"}}' \
SUBJECT_KIND_OVERRIDE_JSON='{"t1_accessory":{"relation":"above_host","layering":"object_contact","max_area":"0.10","min_area":"0.01","dilate":"3","traj":"0.32","subj":"0.05","tar":"8.1"}}' \
ADAPT_OVERRIDE_JSON='{"t1_accessory":{"--relation":"above_host","--mask-layering-mode":"object_contact","--final-edit-mask":"'"$EDIT_MASK"'","--final-edit-mask-mode":"replace","--edit-local-target-prompt":"a small dark red felt top hat with a narrow brim, sitting low on the dog head with a soft natural shadow under the brim, not floating","--edit-local-target-guidance-scale":"0.62","--edit-local-target-cfg-scale":"7.0","--mask-contact-scale":"1.00","--mask-contact-dilate-kernel":"7","--mask-contact-edge-protect-scale":"0.00"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

date
echo fe135_dog_hat_tight_v7_head_contact_sd3_complete
