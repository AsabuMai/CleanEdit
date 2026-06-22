#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH -J pie15-rf2
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie15_rf2_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
hostname; date
echo "### OT-RF ###"
cd "$PROJ/_baselines/src/OT-RF"
"$PROJ/_baselines/envs/ot-rf-otip-py310/bin/python" run_script.py --exp_yaml pie15_exp.yaml || echo "otrf errored"
date; echo "### DRFS ###"
cd "$PROJ/_baselines/src/DeltaRectifiedFlowSampling"
"$PROJ/_baselines/envs/dvrf-py310/bin/python" edit.py --exp_yaml pie15_exp.yaml || echo "drfs errored"
date; echo "### collect ###"
"$PROJ/.venv/bin/python" - <<'PY'
import glob, shutil
from pathlib import Path
from PIL import Image
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
# OT-RF: single output
b=PROJ/"_baselines/src/OT-RF/outputs/OTRF_SD3_pie15/SD3"; n=0
for sd in glob.glob(str(b/"src_*")):
    key=Path(sd).name.replace("src_","",1)
    im=sorted(glob.glob(sd+"/tar_*/output_*.png"))
    if im:
        d=PROJ/"outputs/pie15_baselines_20260618"/key/"ot_rf"/"seed_10"/"result.png"; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy(im[0],d); n+=1
print("collected ot_rf:",n)
# DRFS: concat image -> crop right half
b=PROJ/"_baselines/src/DeltaRectifiedFlowSampling/outputs/DRFS_SD3_pie15/SD3"; n=0
for sd in glob.glob(str(b/"src_*")):
    key=Path(sd).name.replace("src_","",1)
    im=sorted(glob.glob(sd+"/tgt_*/*.png"))
    if im:
        img=Image.open(im[0]).convert("RGB"); w,h=img.size
        crop=img.crop((w//2,0,w,h)) if w>=2*h else img   # right half = edited
        d=PROJ/"outputs/pie15_baselines_20260618"/key/"drfs"/"seed_10"/"result.png"; d.parent.mkdir(parents=True,exist_ok=True); crop.save(d); n+=1
print("collected drfs:",n)
PY
date; echo pie15_rf2_complete
