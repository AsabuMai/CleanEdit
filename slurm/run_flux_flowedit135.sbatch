#!/bin/bash -l
#SBATCH --job-name=cleanedit-flux
#SBATCH --partition=h100
#SBATCH --nodelist=h100-01
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
ROOT=${CLEANEDIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV=${VENV:-$ROOT/.venv}
export MANIFEST=${MANIFEST:-$ROOT/data/flowedit_compatible_135/manifest.local.json}
export FLUX_OUT=${FLUX_OUT:-outputs/flowedit135_flux}
export START=${START:-0}
export LIMIT=${LIMIT:-135}
export SEED=${SEED:-10}
export RUN_PURPOSE=benchmark
export ABLATE=no_final_postprocess
cd "$ROOT"
"$VENV/bin/python" scripts/pie_batch_flux_kindaware_v11.py
