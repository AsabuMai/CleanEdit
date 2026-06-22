#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J flux-knit
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_knit_%j.log

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

# T5 cable-knit: pure-generation texture is weak on FLUX. The procedural knit
# relief (final_knit_texture, gated behind --debug-final-visual-postprocess) is
# the FLUX-port analog to recolor's color blend. Pillows have NO object overlay,
# so the debug flag only triggers the knit relief here (no pasted shape).
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_phase2_dece_flux_paper.py" \
  --true-cfg --distilled-guidance 1.0 --tar-guidance 5.0 \
  --debug-final-visual-postprocess \
  --tasks "pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair" \
  --seeds 10 \
  --output-root outputs/flux_knit_test_20260617

date
echo flux_knit_test_complete
