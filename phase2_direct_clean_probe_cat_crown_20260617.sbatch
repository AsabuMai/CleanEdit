#!/usr/bin/env bash
#SBATCH -J direct_probe_t1
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_direct_clean_probe_cat_crown_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

host="$(hostname)"
case "${host}" in
  a100-01|a100-01.gpu01.cis.k.hosei.ac.jp) ;;
  *) echo "wrong host: ${host}"; exit 2 ;;
esac

export PYTHON="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/../.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODEL_OFFLOAD=0
export LOW_VRAM=0
export NUM_INFERENCE_STEPS=28
export N_MAX=24
export MAX_IMAGE_SIZE=512
export SKIP_EXISTING=0

RUN_TAG="phase2_direct_clean_probe_cat_crown_20260617"
RUN_ID="${RUN_TAG}_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[direct-probe] start=$(date -Is)"
echo "[direct-probe] host=${host}"
echo "[direct-probe] job_id=${SLURM_JOB_ID:-none}"
echo "[direct-probe] run_dir=${RUN_DIR}"
nvidia-smi || true

BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
METHOD_NAME_SUFFIX="_clean_probe" \
TASKS="cat_crown" \
METHODS="direct_target" \
SEEDS="10 11 12" \
bash scripts/run_pretty_matrix.sh

"${PYTHON}" - "${MANIFEST}" <<'PY'
from pathlib import Path
import shlex
import sys

manifest = Path(sys.argv[1])
extra = [
    "--adaptive-clean-control",
    "--adaptive-edit-target-progress", "0.0",
    "--adaptive-edit-target-rms", "0.0",
    "--adaptive-edit-gain", "0.0",
    "--adaptive-preserve-gain", "0.0",
    "--adaptive-edit-weight-min", "1.0",
    "--adaptive-edit-weight-max", "1.0",
    "--adaptive-preserve-weight-min", "1.0",
    "--adaptive-preserve-weight-max", "1.0",
    "--adaptive-projection-scale", "0.0",
    "--adaptive-preserve-clean-correction-scale", "0.0",
    "--adaptive-preserve-drift-budget", "999.0",
]
for raw in manifest.read_text(encoding="utf-8").splitlines():
    path = Path(raw.strip())
    if not path.is_absolute():
        path = (manifest.parent / path).resolve()
    line = path.read_text(encoding="utf-8").strip()
    line = line + " " + " ".join(shlex.quote(x) for x in extra) + "\n"
    path.write_text(line, encoding="utf-8")
    print(f"[direct-probe] patched {path}")
PY

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}" \
  --stop-on-failure

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
method = "direct_target_clean_probe"
for seed in (10, 11, 12):
    p = root / "outputs" / "pretty_matrix" / "cat_crown" / method / f"seed_{seed}" / "stats.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    vals = [r.get("adaptive_edit_target_gap_rms", 0.0) for r in data]
    drift = [r.get("adaptive_preserve_drift", 0.0) for r in data]
    print(
        f"[direct-probe] seed={seed} "
        f"gap_mean={sum(vals)/len(vals):.6f} "
        f"drift_mean={sum(drift)/len(drift):.6f}"
    )
PY

echo "[direct-probe] end=$(date -Is)"
