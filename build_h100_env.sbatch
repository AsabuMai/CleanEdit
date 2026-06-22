#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH -J build-h100-env
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/build_h100_env_%j.log
set -euo pipefail
PROJECT=/cluster/users/grad/2025/25t8103/project
export UV_CACHE_DIR=$PROJECT/.uv-cache
export UV_PYTHON_INSTALL_DIR=$PROJECT/.uv-python
export TMPDIR=$PROJECT/.tmp
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
cd "$PROJECT"
echo host=$(hostname); date
[[ "$(hostname)" == h100-01* ]] || { echo 'not h100-01'; exit 2; }

echo '--- bootstrap pip+uv on py3.9 ---'
/usr/bin/python3 -m venv "$PROJECT/.uv-bootstrap-py39"
"$PROJECT/.uv-bootstrap-py39/bin/pip" install --quiet --upgrade pip
"$PROJECT/.uv-bootstrap-py39/bin/pip" install --quiet uv
UV="$PROJECT/.uv-bootstrap-py39/bin/uv"
"$UV" --version

echo '--- install standalone python 3.12 ---'
"$UV" python install 3.12

echo '--- create venv-h100 (py3.12) ---'
"$UV" venv --python 3.12 "$PROJECT/.venv-h100"

echo '--- install requirements (PyPI + pytorch cu121) ---'
"$UV" pip install --python "$PROJECT/.venv-h100/bin/python"   --index-strategy unsafe-best-match   --extra-index-url https://download.pytorch.org/whl/cu121   -r requirements.txt

echo '--- verify ---'
"$PROJECT/.venv-h100/bin/python" - <<PYEOF
import torch, diffusers, transformers, numpy, scipy, skimage, cv2
print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())
print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)
print('numpy', numpy.__version__)
PYEOF
date; echo build_h100_env_complete
