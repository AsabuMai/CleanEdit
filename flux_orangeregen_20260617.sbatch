#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -J flux-orangeregen
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_orangeregen_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"; mkdir -p logs
host="$(hostname)"; echo "host=$host"; date
[[ "$host" == "a100-01.gpu01.cis.k.hosei.ac.jp" ]] || { echo wrong; exit 2; }
# Regenerate white_bowl_orange (v2 grounded-mask config now in runner) over all
# 3 seeds, overwriting the floating versions in the matrix (no --skip-existing).
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --true-cfg --distilled-guidance 1.0 --tar-guidance 5.0 \
  --tasks white_bowl_orange_tabletop_phase2 --seeds "10 11 12" \
  --output-root outputs/phase2_flux_paper_truecfg_full_20260617
date; echo flux_orangeregen_complete
