import json, os
from PIL import Image, ImageDraw
PROJ="/cluster/users/grad/2025/25t8103/project"
man=json.load(open(f"{PROJ}/data/pie_pilot_20260618/manifest.json"))
OUT=f"{PROJ}/outputs/pie_pilot_20260618"
sub=200; gap=4; lh=26
# group by family order T1..T5, each row = [src|result] x3 entries
fams=["T1","T2","T3","T4","T5"]
by={f:[] for f in fams}
for e in man: by[e["family"]].append(e)
rowh=sub+lh
cellw=sub*2+gap
canvas=Image.new("RGB",(cellw*3+20, rowh*5+20),(255,255,255))
for ri,f in enumerate(fams):
    for ci,e in enumerate(by[f][:3]):
        src=Image.open(e["image"]).convert("RGB").resize((sub,sub))
        rp=f"{OUT}/{e['key']}/dece_rf_flux/seed_10/result.png"
        res=Image.open(rp).convert("RGB").resize((sub,sub)) if os.path.exists(rp) else Image.new("RGB",(sub,sub),(50,0,0))
        cell=Image.new("RGB",(cellw,rowh),(255,255,255))
        cell.paste(src,(0,lh)); cell.paste(res,(sub+gap,lh))
        tgt=e["target_prompt"]
        ImageDraw.Draw(cell).text((2,2),f"{f} {e['key'].split('_',1)[1]}: {tgt[:34]}",fill=(0,0,0))
        canvas.paste(cell,(ci*(cellw+6)+6, ri*(rowh+2)+6))
canvas.save("/tmp/pie_contact.png"); print("saved",canvas.size)
