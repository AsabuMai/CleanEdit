import os, glob, shutil
from pathlib import Path
from PIL import Image
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
OUTROOT=PROJ/os.environ["OUTROOT"]; SRC=PROJ/"_baselines/src"
SPECS=[("flowedit_sd3",SRC/"FlowEdit/outputs/FlowEdit_SD3_pie15/SD3","tar",False),
       ("splitflow_sd3",SRC/"SplitFlow/outputs/SplitFlow_SD3_pie15/SD3","tar",False),
       ("ot_rf",SRC/"OT-RF/outputs/OTRF_SD3_pie15/SD3","tar",False),
       ("drfs",SRC/"DeltaRectifiedFlowSampling/outputs/DRFS_SD3_pie15/SD3","tgt",True)]
for method,base,sub,concat in SPECS:
    n=0
    for sd in glob.glob(str(base/"src_*")):
        key=Path(sd).name.replace("src_","",1)
        imgs=sorted(glob.glob(sd+f"/{sub}_*/*.png"))
        if not imgs: continue
        dst=OUTROOT/key/method/"seed_10"/"result.png"; dst.parent.mkdir(parents=True,exist_ok=True)
        if concat:
            im=Image.open(imgs[0]).convert("RGB"); w,h=im.size
            (im.crop((w//2,0,w,h)) if w>=2*h else im).save(dst)
        else: shutil.copy(imgs[0],dst)
        n+=1
    print("collected",method,n)
