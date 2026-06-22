import json, os
from PIL import Image, ImageDraw
PROJ="/cluster/users/grad/2025/25t8103/project"
man=json.load(open(f"{PROJ}/data/pie_pilot_20260618/manifest.json"))
FLUX=f"{PROJ}/outputs/pie_auto_flux_20260618"; SD3=f"{PROJ}/outputs/pie_auto_sd3_20260618"
sub=180; gap=3; lh=24
def load(p): return Image.open(p).convert("RGB").resize((sub,sub)) if os.path.exists(p) else Image.new("RGB",(sub,sub),(50,0,0))
# one page per family: 3 entries rows x [src|flux|sd3] cols
for fam in ["T1","T2","T3","T4","T5"]:
    es=[e for e in man if e["family"]==fam][:3]
    cellw=sub*3+gap*2
    canvas=Image.new("RGB",(cellw+12, (sub+lh+6)*len(es)+24),(255,255,255))
    d0=ImageDraw.Draw(canvas); d0.text((4,4),f"{fam}   [ source | FLUX | SD3 ]",fill=(0,0,0))
    for ri,e in enumerate(es):
        src=load(e["image"])
        fl=load(f"{FLUX}/{e['key']}/dece_rf_flux/seed_10/result.png")
        sd=load(f"{SD3}/{e['key']}/support_v3_controller_rmsgap/seed_10/result.png")
        y=20+ri*(sub+lh+6)
        row=Image.new("RGB",(cellw,sub+lh),(255,255,255))
        row.paste(src,(0,lh)); row.paste(fl,(sub+gap,lh)); row.paste(sd,(2*(sub+gap),lh))
        ImageDraw.Draw(row).text((2,2),e["target_prompt"][:60],fill=(0,0,0))
        canvas.paste(row,(6,y))
    canvas.save(f"/tmp/pievs_{fam}.png"); print("saved",fam,canvas.size)
