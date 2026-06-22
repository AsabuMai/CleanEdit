#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J fe135-doghat-v5
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_dog_hat_tight_v5_v2mask_softdark_sd3_%j.log

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

MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_dog_hat_tightmask_v5_v2mask_softdark.json \
OUT=outputs/fe135_dog_hat_tight_v5_v2mask_softdark_sd3 \
SUBJECT_PRESERVE_MODE=1 \
SUBJECT_OBJECT_PROVIDER=operation_support_v3 \
SUBJECT_GROUNDING_METHOD=external_mask \
KIND_OVERRIDE_JSON='{"t1_accessory":{"hedit":"1.12","text":"0.13","rec":"0.21","struct":"0.42","core":"0.95","subj":"0.05"}}' \
SUBJECT_KIND_OVERRIDE_JSON='{"t1_accessory":{"relation":"above_host","layering":"object_contact","max_area":"0.12","min_area":"0.01","dilate":"3","traj":"0.34","subj":"0.05","tar":"7.8"}}' \
ADAPT_OVERRIDE_JSON='{"t1_accessory":{"--relation":"above_host","--mask-layering-mode":"object_contact","--final-edit-mask-mode":"replace","--edit-local-target-prompt":"a small dark red felt top hat with a short cylindrical crown and visible narrow brim resting naturally on the dog head","--edit-local-target-guidance-scale":"0.56","--edit-local-target-cfg-scale":"6.8","--mask-contact-scale":"0.85","--mask-contact-dilate-kernel":"7","--mask-contact-edge-protect-scale":"0.10"}}' \
"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

date
echo fe135_dog_hat_tight_v5_v2mask_softdark_sd3_complete
