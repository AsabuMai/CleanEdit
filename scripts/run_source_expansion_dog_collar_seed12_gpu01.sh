#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J src25-dog
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/source_expansion_dog_collar_seed12_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/source_expansion_dog_collar_seed12_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
date
nvidia-smi || true

if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run source expansion generation outside a100-01" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ALLOW_MASK_DOWNLOAD=1
export RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1
export SKIP_EXISTING=0
export REUSE_SEMANTIC_MASKS=1
export REGENERATE_MASKS=0

echo "== source expansion dog collar replacement, DeCE-RF seed12 =="
TASKS="web_dog_blue_collar" \
METHODS="support_v3_controller_rmsgap" \
SEEDS="12" \
bash scripts/run_pretty_matrix.sh

.venv/bin/python - <<'PY'
from pathlib import Path
result = Path("outputs/pretty_matrix/web_dog_blue_collar/support_v3_controller_rmsgap/seed_12/result.png")
print("web_dog_blue_collar", "ok" if result.exists() else "missing", result)
if not result.exists():
    raise SystemExit("missing web_dog_blue_collar")
PY

date
