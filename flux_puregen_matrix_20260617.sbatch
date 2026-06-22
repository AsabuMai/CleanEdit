#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH -J flux-puregen
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_puregen_matrix_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo "Refusing outside a100-01"; exit 2; }
# PURE-GENERATION matrix: object overlays disabled globally (only material knit
# relief + recolor blend kept), true-CFG, per-task fixes (sunglasses pct20, bow
# pct25+edit_region0, orange v2 mask, knit 6.0). 15 tasks x seeds 10/11/12.
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --true-cfg --distilled-guidance 1.0 --tar-guidance 5.0 \
  --seeds "10 11 12" --skip-existing \
  --output-root outputs/phase2_flux_paper_puregen_20260617
date; echo flux_puregen_matrix_complete
