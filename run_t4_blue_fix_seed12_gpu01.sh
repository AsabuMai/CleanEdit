#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J t4-blue-s12
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/t4_blue_fix_seed12_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/t4_blue_fix_seed12_%j.err

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

TASKS="web_red_suitcase_to_blue web_orange_handbag_to_blue"

export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ALLOW_MASK_DOWNLOAD=1
export RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1
export SKIP_EXISTING=0
export REUSE_SEMANTIC_MASKS=0
export REGENERATE_MASKS=1

echo "== T4 stronger blue validation: seed 12 =="
TASKS="$TASKS" \
METHODS="support_v3_controller_rmsgap" \
SEEDS="12" \
bash scripts/run_pretty_matrix.sh

echo "== coverage =="
.venv/bin/python - <<'PY'
from pathlib import Path

tasks = "web_red_suitcase_to_blue web_orange_handbag_to_blue".split()
root = Path("outputs/pretty_matrix")
missing = []
for task in tasks:
    result = root / task / "support_v3_controller_rmsgap" / "seed_12" / "result.png"
    meta = root / task / "support_v3_controller_rmsgap" / "seed_12" / "metadata.json"
    print(task, "result", "ok" if result.exists() else "missing", result)
    print(task, "metadata", "ok" if meta.exists() else "missing", meta)
    if not result.exists() or not meta.exists():
        missing.append(task)
if missing:
    raise SystemExit("missing outputs: " + ", ".join(missing))
PY

date
