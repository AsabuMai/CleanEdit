#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=18:00:00
#SBATCH -J sam-sd3-s10
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/samflow_sd3_seed10_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/samflow_sd3_seed10_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
EXP="$PROJECT/experiments/support_v3_2026-06-02"
TASKS_T1_T4="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"
TASKS_T5="pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"
SEEDS="10"
SAM_BASELINES="sam_flow_sd3"

export HF_HOME="$PROJECT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PROJECT/.cache/huggingface/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT"
mkdir -p "$SRC" "$BASE/logs"

host="$(hostname)"
echo "host=$host"
date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

fetch_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local path="$SRC/$name"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi
  git -C "$path" fetch --all --tags --prune
  git -C "$path" checkout "$commit"
  git -C "$path" rev-parse HEAD
}

echo "== fetch Sam-Flow =="
fetch_repo "Sam-Flow" "https://github.com/chwbob/Sam-Flow.git" "f5f70c9dcc77b0dc6b90c29bb863452128850cb5"

echo "== prepare Sam-Flow env =="
python3 scripts/prepare_e2_baseline_envs.py --only Sam-Flow --recreate --timeout-s 3600

echo "== smoke Sam-Flow entrypoint =="
python3 scripts/smoke_baseline_entrypoints.py --only Sam-Flow

echo "== ensure Sam-Flow manifest rows =="
.venv/bin/python scripts/ensure_samflow_manifest_rows.py --scope all --baselines "$SAM_BASELINES" --seeds "10 11 12"

echo "== run Sam-Flow seed10 T1-T4 =="
"$BASE/envs/sam-flow-py310/bin/python" scripts/run_samflow_baseline.py \
  --manifest "$EXP/e2_t1_t4_formal_baseline_manifest.csv" \
  --samflow-root "$SRC/Sam-Flow" \
  --python "$BASE/envs/sam-flow-py310/bin/python" \
  --baselines "$SAM_BASELINES" \
  --tasks "$TASKS_T1_T4" \
  --seeds "$SEEDS" \
  --max-image-size 512 \
  --skip-complete

echo "== run Sam-Flow seed10 T5 =="
"$BASE/envs/sam-flow-py310/bin/python" scripts/run_samflow_baseline.py \
  --manifest "$EXP/e2_t5_formal_baseline_manifest.csv" \
  --samflow-root "$SRC/Sam-Flow" \
  --python "$BASE/envs/sam-flow-py310/bin/python" \
  --baselines "$SAM_BASELINES" \
  --tasks "$TASKS_T5" \
  --seeds "$SEEDS" \
  --max-image-size 512 \
  --skip-complete

echo "== seed10 preview sheet =="
.venv/bin/python scripts/make_samflow_seed10_preview.py

echo "== seed10 coverage =="
.venv/bin/python - <<'PY'
from pathlib import Path
tasks = "cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2 pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair".split()
methods = ["sam_flow_sd3", "sam_flow_flux"]
missing = []
for method in methods:
    for task in tasks:
        path = Path("outputs") / "baselines" / method / task / "seed_10" / "result.png"
        if not path.exists():
            missing.append(str(path))
print("expected", len(tasks) * len(methods), "missing", len(missing))
for item in missing[:30]:
    print("missing", item)
if missing:
    raise SystemExit(1)
PY

date
echo samflow_seed10_preview_complete
