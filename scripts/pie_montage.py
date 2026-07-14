import json, os
from PIL import Image, ImageDraw
PROJ="/cluster/users/grad/2025/25t8103/project"
man={e["key"]:e for e in json.load(open(f"{PROJ}/data/pie_pilot_20260618/manifest.json"))}
B=f"{PROJ}/outputs/pie15_baselines_20260618"
METH=[("source",None,None),("CleanEdit-FLUX",f"{PROJ}/outputs/pie_auto_flux_20260618","dece_rf_flux"),
 ("CleanEdit-SD3",f"{PROJ}/outputs/pie_auto_sd3_20260618","support_v3_controller_rmsgap"),
 ("FireFlow",B,"fireflow"),("RF-Solver",B,"rf_solver_edit"),("ReFlex",B,"reflex"),
 ("SamFlow-F",B,"sam_flow_flux"),("SamFlow-S",B,"sam_flow_sd3"),("FlowEdit",B,"flowedit_sd3"),
 ("SplitFlow",B,"splitflow_sd3"),("OT-RF",B,"ot_rf"),("DRFS",B,"drfs")]
keys=["T1_2_18","T2_2_68","T3_2_01","T4_6_14","T5_7_00"]
c=140; lh=14
W=len(METH)*(c+2); H=len(keys)*(c+lh+2)+16
canvas=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(canvas)
for ci,(name,dd,sub) in enumerate(METH): d.text((ci*(c+2)+2,2),name,fill=(0,0,0))
for ri,k in enumerate(keys):
    y=16+ri*(c+lh+2)
    for ci,(name,dd,sub) in enumerate(METH):
        if name=="source": p=f"{PROJ}/{man[k]['image']}"
        else: p=f"{dd}/{k}/{sub}/seed_10/result.png"
        try: im=Image.open(p).convert("RGB").resize((c,c))
        except: im=Image.new("RGB",(c,c),(60,0,0))
        canvas.paste(im,(ci*(c+2),y+lh))
        if ci==0: d.text((2,y+2),k,fill=(0,0,0))
canvas.save("/tmp/pie_montage.png"); print("saved",canvas.size)
