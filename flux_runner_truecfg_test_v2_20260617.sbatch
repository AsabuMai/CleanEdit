#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J flux-runner-tc
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_runner_truecfg_%j.log

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

# Run the OFFICIAL faithful runner (full per-task config: custom masks,
# attention-derived core, edit-region/target/local-target, anti-tiling prompts)
# with the true-CFG fix injected (--true-cfg --distilled-guidance 1.0) and
# tar-guidance 5.0. First validate the two problem decal tasks.
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --true-cfg --distilled-guidance 1.0 --tar-guidance 5.0 \
  --tasks "mug_heart tshirt_star" --seeds 10 \
  --output-root outputs/phase2_flux_truecfg_test_v2_20260617

date
echo flux_runner_truecfg_test_complete
