from PIL import Image, ImageDraw
import os
old="outputs/phase2_flux_paper_final_20260618"
new="outputs/flux_knitfill_20260618"
rows=[]
for t,tag in [("pillow_same_color_cable_knit","white"),("pillow_same_color_cable_knit_armchair","armchair")]:
    o=Image.open(os.path.join(old,t,"dece_rf_flux","seed_10","result.png")).convert("RGB")
    n=Image.open(os.path.join(new,t,"dece_rf_flux","seed_10","result.png")).convert("RGB")
    W,H=o.size; sz=(300,int(300*H/W))
    o=o.resize(sz); n=n.resize(sz)
    row=Image.new("RGB",(sz[0]*2+8, sz[1]+18),(255,255,255))
    row.paste(o,(0,18)); row.paste(n,(sz[0]+8,18))
    ImageDraw.Draw(row).text((3,4),tag+"   OLD(center strip) | NEW(wide core)",fill=(0,0,0))
    rows.append(row)
W=max(r.width for r in rows); H=sum(r.height for r in rows)+8
c=Image.new("RGB",(W,H),(255,255,255)); y=0
for r in rows: c.paste(r,(0,y)); y+=r.height+8
c.save("/tmp/knitfill_cmp.png"); print("saved",c.size)
