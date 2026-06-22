#!/usr/bin/env bash
#SBATCH -J p2_mugred
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t3_mug_redlock_sweep_3seed_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t3_mug_redlock_sweep_3seed_20260611_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
ARTIFACT="${ROOT}/experiments/support_v3_2026-06-02/phase2_t3_mug_redlock_sweep_3seed_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[p2-mug-redlock] start=$(date -Is)"
echo "[p2-mug-redlock] host=$(hostname)"
echo "[p2-mug-redlock] job_id=${SLURM_JOB_ID:-none}"
nvidia-smi || true

export PYTHON="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/../.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODEL_OFFLOAD=0
export LOW_VRAM=0
export NUM_INFERENCE_STEPS=28
export N_MAX=24
export MAX_IMAGE_SIZE=512
export REGENERATE_MASKS=1
export REUSE_SEMANTIC_MASKS=0

PROMPT="A minimalist photo of the same white ceramic mug with a small flat hard-edged solid red heart decal printed cleanly on the front, with sharp printed edges and no glow or blur, while the mug shape, handle, highlights, bottom shadow, grey background, and lighting remain unchanged."

add_variant() {
  local suffix="$1"
  local box="$2"
  local cdelta="$3"
  BATCH_MANIFEST="${MANIFEST}" \
  PREPARE_ONLY=1 \
  SKIP_EXISTING=0 \
  TASKS="mug_heart" \
  METHODS="support_v3_controller_rmsgap" \
  SEEDS="10 11 12" \
  METHOD_NAME_SUFFIX="${suffix}" \
  TARGET_PROMPT_OVERRIDE="${PROMPT}" \
  DECAL_BOX_OVERRIDE="${box}" \
  SUPPORT_V3_LOCALIZED_DECAL_REF_GUIDANCE_SCALE="0.65" \
  SUPPORT_V3_LOCALIZED_DECAL_REF_LUMA_PRESERVE="0.00" \
  SUPPORT_V3_LOCALIZED_DECAL_REF_GRADIENT_PRESERVE="0.00" \
  SUPPORT_V3_LOCALIZED_DECAL_REF_SMOOTH_KERNEL=0 \
  TASK_FINAL_REF_COMPOSITE_SCALE="0.35" \
  TASK_FINAL_REF_COMPOSITE_BLUR="0.0" \
  SUPPORT_V3_OPFIELD_DECAL_REC="0.12" \
  SUPPORT_V3_OPFIELD_DECAL_TRAJ="0.04" \
  SUPPORT_V3_OPFIELD_DECAL_HEDIT="0.72" \
  SUPPORT_V3_OPFIELD_DECAL_TEXT="0.12" \
  COMPLETION_CLEAN_DELTA_SCALE="${cdelta}" \
  COMPLETION_CLEAN_DELTA_SCHEDULE_START="0.0" \
  COMPLETION_CLEAN_DELTA_SCHEDULE_STOP="0.0" \
  bash scripts/run_pretty_matrix.sh
}

add_variant "_mugred_c180_v1" "0.390,0.415,0.535,0.570" "1.80"
add_variant "_mugred_c160_v1" "0.395,0.418,0.532,0.568" "1.60"
add_variant "_mugred_c140_v1" "0.398,0.420,0.528,0.565" "1.40"

echo "[p2-mug-redlock] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
from pathlib import Path
root = Path("/cluster/users/grad/2025/25t8103/project")
methods = [
    "support_v3_controller_rmsgap_mugred_c180_v1",
    "support_v3_controller_rmsgap_mugred_c160_v1",
    "support_v3_controller_rmsgap_mugred_c140_v1",
]
missing = []
for method in methods:
    for seed in ("10", "11", "12"):
        d = root / "outputs" / "pretty_matrix" / "mug_heart" / method / f"seed_{seed}"
        absent = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
        if absent:
            missing.append({"method": method, "seed": seed, "missing": absent})
print(f"[p2-mug-redlock] missing={missing}")
if missing:
    raise SystemExit(1)
PY

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}" \
  "outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap_mugred_c180_v1" \
  "outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap_mugred_c160_v1" \
  "outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap_mugred_c140_v1"

echo "[p2-mug-redlock] artifact=${ARTIFACT}"
echo "[p2-mug-redlock] end=$(date -Is)"
