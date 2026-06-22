#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=03:00:00
#SBATCH -J t5-shirt-s1012
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/t5_shirt_fix_seeds10_12_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/t5_shirt_fix_seeds10_12_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
date
nvidia-smi || true

if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run generation outside a100-01" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ALLOW_MASK_DOWNLOAD=1
export RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1
export SKIP_EXISTING=0
export REUSE_SEMANTIC_MASKS=0
export REGENERATE_MASKS=1

echo "== T5 shirt waffle-panel validation: seeds 10,11,12 =="
TASKS="web_white_shirt_lace_panel" \
METHODS="support_v3_controller_rmsgap" \
SEEDS="10 11 12" \
bash scripts/run_pretty_matrix.sh

.venv/bin/python - <<'PY'
from pathlib import Path

root = Path("outputs/pretty_matrix/web_white_shirt_lace_panel/support_v3_controller_rmsgap")
missing = []
for seed in (10, 11, 12):
    for name in ("result.png", "metadata.json"):
        path = root / f"seed_{seed}" / name
        print(seed, name, "ok" if path.exists() else "missing")
        if not path.exists():
            missing.append(str(path))
if missing:
    raise SystemExit("missing outputs: " + ", ".join(missing))
PY

date
