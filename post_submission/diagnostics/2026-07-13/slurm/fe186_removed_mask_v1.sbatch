#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -J fe186-mask
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe186_removed_mask_v1_%j.log

set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
cd "$PROJ"
test "$(hostname)" = "a100-01.gpu01.cis.k.hosei.ac.jp"
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export MANIFEST=$PROJ/tmp/fe186_manifest.json
export OUT_DIR=$PROJ/tmp/fe186_removed_mask_v1
export OUT_MANIFEST=$PROJ/tmp/fe186_removed_mask_v1_manifest.json
export KEEP_ALL_BOXES=1
export BOX_SCORE_FLOOR=0.18
export BOX_SCORE_REL=0.3
export BOX_APPEARANCE_DISTANCE=0.30
export T2_REPLACE_WITH_REMOVED=1
export REMOVED_MAX_AREA=0.18

"$PROJ/.venv/bin/python" scripts/make_flowedit118_sam_support_manifest.py
