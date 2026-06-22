#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH -J pie15-splitflow
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie15_splitflow_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
cd "$PROJ/_baselines/src/SplitFlow"
"$PROJ/_baselines/envs/splitflow-py310/bin/python" run_script.py --exp_yaml pie15_exp.yaml || echo "splitflow errored"
echo "=== collect ==="
"$PROJ/.venv/bin/python" - <<'PY'
import glob, shutil
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
base=PROJ/"_baselines/src/SplitFlow/outputs/SplitFlow_SD3_pie15/SD3"
n=0
for srcdir in glob.glob(str(base/"src_*/tar_0")):
    key=Path(srcdir).parent.name.replace("src_","",1)
    imgs=sorted(glob.glob(srcdir+"/output_*.png"))
    if imgs:
        dst=PROJ/"outputs/pie15_baselines_20260618"/key/"splitflow_sd3"/"seed_10"/"result.png"
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy(imgs[0],dst); n+=1
print("collected splitflow_sd3:",n)
PY
date; echo pie15_splitflow_complete
