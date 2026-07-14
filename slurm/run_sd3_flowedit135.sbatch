#!/bin/bash -l
#SBATCH --job-name=cleanedit-sd3
#SBATCH --partition=a100
#SBATCH --nodelist=a100-01
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
ROOT=${CLEANEDIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV=${VENV:-$ROOT/.venv}
export MANIFEST=${MANIFEST:-$ROOT/data/flowedit_compatible_135/manifest.local.json}
export OUT=${OUT:-outputs/flowedit135_sd3}
export START=${START:-0}
export LIMIT=${LIMIT:-135}
export SEED=${SEED:-10}
export ABLATE=no_final_postprocess
cd "$ROOT"
"$VENV/bin/python" scripts/pie_sd3_batch_kindaware.py
