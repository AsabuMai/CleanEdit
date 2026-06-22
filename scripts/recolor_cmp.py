import json, os
from PIL import Image, ImageDraw
base = "outputs/phase2_flux_paper_final_20260618"
items = [("green_mug_orange_phase2","mug  green->orange"),
         ("red_office_chair_to_blue_office_chair","chair  red->blue"),
         ("yellow_vase_blue_phase2","vase  yellow->blue")]
sub=300; gap=6; lh=20
rows=[]
for t,lbl in items:
    rd=os.path.join(base,t,"dece_rf_flux","seed_10")
    res=os.path.join(rd,"result.png"); meta=os.path.join(rd,"metadata.json")
    src=None
    if os.path.exists(meta):
        m=json.load(open(meta))
        for k in ("source_image","source","src_image","image"):
            if m.get(k): src=m[k]; break
    row=Image.new("RGB",(sub*2+gap, sub+lh),(255,255,255))
    for j,p in enumerate((src,res)):
        im=Image.open(p).convert("RGB").resize((sub,sub)) if (p and os.path.exists(p)) else Image.new("RGB",(sub,sub),(40,40,40))
        row.paste(im,(j*(sub+gap),lh))
    ImageDraw.Draw(row).text((3,4),lbl+"   [source | result]",fill=(0,0,0))
    rows.append(row)
W=sub*2+gap; H=sum(r.height for r in rows)+ (len(rows)-1)*8
c=Image.new("RGB",(W,H),(255,255,255)); y=0
for r in rows: c.paste(r,(0,y)); y+=r.height+8
c.save("/tmp/recolor_cmp.png"); print("saved", c.size)
