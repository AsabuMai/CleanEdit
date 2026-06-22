from PIL import Image, ImageDraw
import os
base="outputs/phase2_flux_paper_final_20260618"
items=[("pillow_same_color_cable_knit_grey","grey (unchanged)"),
       ("pillow_same_color_cable_knit","white (fixed)"),
       ("pillow_same_color_cable_knit_armchair","armchair (fixed)")]
cells=[]
for t,lbl in items:
    im=Image.open(os.path.join(base,t,"dece_rf_flux","seed_10","result.png")).convert("RGB")
    W,H=im.size; im=im.resize((300,int(300*H/W)))
    cell=Image.new("RGB",(300,im.height+18),(255,255,255))
    cell.paste(im,(0,18)); ImageDraw.Draw(cell).text((3,4),lbl,fill=(0,0,0))
    cells.append(cell)
W=sum(c.width for c in cells)+16; H=max(c.height for c in cells)
canvas=Image.new("RGB",(W,H),(255,255,255)); x=0
for c in cells: canvas.paste(c,(x,0)); x+=c.width+8
canvas.save("/tmp/knit_final_show.png"); print("saved",canvas.size)
