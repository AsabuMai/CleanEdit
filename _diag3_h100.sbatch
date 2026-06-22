#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH -J diag3-h100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/diag3_h100_%j.log
echo host=$(hostname)
echo '--- pypi reachable from h100 node? ---'
/usr/bin/python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=8); print('PYPI_OK')" 2>&1 | tail -1
echo '--- pytorch index reachable? ---'
/usr/bin/python3 -c "import urllib.request; urllib.request.urlopen('https://download.pytorch.org/whl/cu121', timeout=8); print('TORCH_IDX_OK')" 2>&1 | tail -1
echo '--- pip available for py3.9? ---'; /usr/bin/python3 -m pip --version 2>&1 | tail -1
echo '--- disk space on project fs ---'; df -h /cluster/users/grad/2025/25t8103/project | tail -1
