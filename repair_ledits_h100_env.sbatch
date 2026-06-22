#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J repair-ledits-h100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/repair_ledits_h100_env_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
cd "$PROJ"
hostname
date
if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside h100-01" >&2
  exit 2
fi
ENV=$PROJ/_baselines/envs-h100/ledits-pp-py39
"$ENV/bin/python" -m pip install --no-build-isolation -e "$PROJ/_baselines/src/ledits_pp"
"$ENV/bin/python" - <<'PY'
import torch
import diffusers
import leditspp
from diffusers import LEditsPPPipelineStableDiffusion
print("ledits repair ok", torch.__version__, torch.cuda.is_available(), diffusers.__version__)
PY
date
echo repair_ledits_h100_env_complete
