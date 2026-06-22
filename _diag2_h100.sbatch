#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH -J diag2-h100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/diag2_h100_%j.log
echo host=$(hostname)
echo '--- module system? ---'; (module avail) 2>&1 | head -40 || echo 'no module cmd'
echo '--- conda? ---'; which conda 2>&1; ls -d /opt/conda* /opt/python* 2>&1
echo '--- any python3.1x on PATH/usr ---'; ls /usr/bin/python3.* /usr/local/bin/python3.* 2>&1
echo '--- find python3.12 (fast dirs) ---'; ls /opt 2>&1; find /usr /opt -maxdepth 3 -name 'python3.1[0-9]' 2>/dev/null | head
