#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH -J diag4-h100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/diag4_h100_%j.log
echo host=$(hostname)
t () { /usr/bin/python3 -c "import urllib.request as u; u.urlopen('$1', timeout=8); print('OK  $1')" 2>&1 | tail -1; }
t https://github.com
t https://astral.sh/uv/install.sh
t https://objects.githubusercontent.com
t https://raw.githubusercontent.com
t https://files.pythonhosted.org
echo '--- ensurepip for py3.9 ---'
/usr/bin/python3 -m ensurepip --version 2>&1 | tail -1
