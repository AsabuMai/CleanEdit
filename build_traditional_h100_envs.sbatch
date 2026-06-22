#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH -J build-trad-h100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/build_traditional_h100_envs_%j.log
set -euo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
ENV_ROOT=$PROJ/_baselines/envs-h100
export PIP_CACHE_DIR=$PROJ/.pip_cache
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
hostname
date
cd "$PROJ"
if [[ "$(hostname)" != "h100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to build outside h100-01" >&2
  exit 2
fi
mkdir -p "$ENV_ROOT" "$PIP_CACHE_DIR"

build_ip2p() {
  local env=$ENV_ROOT/instruct-pix2pix-py39
  rm -rf "$env"
  /usr/bin/python3 -m venv "$env"
  "$env/bin/python" -m pip install -U pip "setuptools<81" wheel
  "$env/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
  "$env/bin/python" -m pip install \
    diffusers==0.21.4 transformers==4.33.3 accelerate==0.23.0 \
    huggingface-hub==0.25.2 tokenizers==0.13.3 safetensors==0.4.5 \
    pillow==10.4.0 numpy==1.26.4 scipy==1.13.1 tqdm==4.67.1
  "$env/bin/python" - <<'PY'
import torch, diffusers
from diffusers import StableDiffusionInstructPix2PixPipeline
print("ip2p env ok", torch.__version__, torch.cuda.is_available(), diffusers.__version__)
PY
}

build_ledits() {
  local env=$ENV_ROOT/ledits-pp-py39
  rm -rf "$env"
  /usr/bin/python3 -m venv "$env"
  "$env/bin/python" -m pip install -U pip "setuptools<81" wheel
  "$env/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
  "$env/bin/python" -m pip install \
    diffusers==0.20.2 transformers==4.33.3 accelerate==0.23.0 \
    huggingface-hub==0.25.2 tokenizers==0.13.3 safetensors==0.4.5 \
    sentencepiece==0.2.0 ftfy==6.2.3 pillow==10.4.0 numpy==1.26.4 scipy==1.13.1 tqdm==4.67.1
  "$env/bin/python" -m pip install -e "$PROJ/_baselines/src/ledits_pp"
  "$env/bin/python" - <<'PY'
import torch, diffusers, leditspp
from diffusers import LEditsPPPipelineStableDiffusion
print("ledits env ok", torch.__version__, torch.cuda.is_available(), diffusers.__version__)
PY
}

build_ip2p
build_ledits
date
echo build_traditional_h100_envs_complete
