#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -J fe135-t4-tex-sd3
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_t4_texture_sd3_%j.log

set -euo pipefail

PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MANIFEST=$PROJ/data/flowedit_compatible_135/manifest_t4_recolor_19.json
export OUT=outputs/fe135_t4_texture_sd3
export SEED=10
export T4_TEXTURE_DETAIL_TRANSFER=1
export T4_TEXTURE_DETAIL_TRANSFER_STRENGTH=0.8
export T4_TEXTURE_DETAIL_TRANSFER_KERNEL=9

hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

"$PROJ/.venv/bin/python" scripts/pie_sd3_batch_kindaware.py

echo "[count]"
find "$PROJ/$OUT" -path "*/seed_${SEED}/result.png" | wc -l
date
echo fe135_t4_texture_sd3_complete
