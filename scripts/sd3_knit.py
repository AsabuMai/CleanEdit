from PIL import Image, ImageDraw
import os
base="outputs/pretty_matrix"; m="support_v3_controller_rmsgap"
items=[("pillow_same_color_cable_knit_grey","SD3 grey"),
       ("pillow_same_color_cable_knit","SD3 white"),
       ("pillow_same_color_cable_knit_armchair","SD3 armchair")]
cells=[]
for t,lbl in items:
    p=os.path.join(base,t,m,"seed_10","result.png")
    if not os.path.exists(p):
        print("MISSING",p); continue
    im=Image.open(p).convert("RGB"); W,H=im.size; im=im.resize((300,int(300*H/W)))
    cell=Image.new("RGB",(300,im.height+18),(255,255,255)); cell.paste(im,(0,18))
    ImageDraw.Draw(cell).text((3,4),lbl,fill=(0,0,0)); cells.append(cell)
W=sum(c.width for c in cells)+16; H=max(c.height for c in cells)
canvas=Image.new("RGB",(W,H),(255,255,255)); x=0
for c in cells: canvas.paste(c,(x,0)); x+=c.width+8
canvas.save("/tmp/sd3_knit.png"); print("saved",canvas.size)
