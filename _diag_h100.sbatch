#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH -J diag-h100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/diag_h100_%j.log
cd /cluster/users/grad/2025/25t8103/project
echo host=$(hostname)
echo '--- which python3 + version ---'
which python3; python3 --version
echo '--- python3.12 path ---'; ls -la /usr/bin/python3.12 2>&1
echo '--- venv python version ---'; .venv/bin/python --version 2>&1
echo '--- venv import torch ---'; .venv/bin/python -c 'import torch; print(torch.__version__)' 2>&1 | tail -3
echo '--- venv site-packages dirs ---'; ls .venv/lib 2>&1
