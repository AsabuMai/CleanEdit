#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=08:00:00
#SBATCH -J src25-s1011
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/source_expansion_final10_decerf_seeds10_11_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/source_expansion_final10_decerf_seeds10_11_%j.err

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

TASKS="web_woman_black_sunglasses web_cat_collar_bell web_plate_red_apple web_wicker_basket_apples web_mug_red_heart_decal web_notebook_heart_sticker web_red_suitcase_to_blue web_orange_handbag_to_blue web_gray_hoodie_quilted_panel web_white_shirt_lace_panel"

export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ALLOW_MASK_DOWNLOAD=1
export RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1
export SKIP_EXISTING=1
export REUSE_SEMANTIC_MASKS=1
export REGENERATE_MASKS=0

echo "== final 10 source expansion DeCE-RF seeds 10,11 =="
TASKS="$TASKS" \
METHODS="support_v3_controller_rmsgap" \
SEEDS="10 11" \
bash scripts/run_pretty_matrix.sh

echo "== coverage =="
.venv/bin/python - <<'PY'
from pathlib import Path

tasks = "web_woman_black_sunglasses web_cat_collar_bell web_plate_red_apple web_wicker_basket_apples web_mug_red_heart_decal web_notebook_heart_sticker web_red_suitcase_to_blue web_orange_handbag_to_blue web_gray_hoodie_quilted_panel web_white_shirt_lace_panel".split()
root = Path("outputs/pretty_matrix")
missing = []
for task in tasks:
    for seed in (10, 11, 12):
        result = root / task / "support_v3_controller_rmsgap" / f"seed_{seed}" / "result.png"
        print(task, seed, "ok" if result.exists() else "missing", result)
        if not result.exists():
            missing.append((task, seed))
if missing:
    raise SystemExit("missing results: " + ", ".join(f"{t}/seed_{s}" for t, s in missing))
PY

date
