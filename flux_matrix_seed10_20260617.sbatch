#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -J flux-matrix-s10
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_matrix_seed10_%j.log

set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT"
mkdir -p logs

host="$(hostname)"
echo "host=$host"; date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2; exit 2
fi

# Full 15-task matrix, seed 10, official faithful runner config + true-CFG fix
# (--true-cfg --distilled-guidance 1.0) at tar-guidance 5.0. Global baseline pass.
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --true-cfg --distilled-guidance 1.0 --tar-guidance 5.0 \
  --seeds 10 --skip-existing \
  --output-root outputs/phase2_flux_paper_truecfg_seed10_20260617

date
echo flux_matrix_seed10_complete
