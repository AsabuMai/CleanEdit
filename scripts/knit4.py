from PIL import Image, ImageDraw
import os
def z(p):
    im=Image.open(p).convert("RGB"); W,H=im.size
    return im.crop((int(W*0.18),int(H*0.18),int(W*0.82),int(H*0.82))).resize((300,300))
panels=[
 ("SD3 target (all-over)","outputs/pretty_matrix/pillow_same_color_cable_knit/support_v3_controller_rmsgap/seed_10/result.png"),
 ("FLUX current (chunky pct8)","outputs/phase2_flux_paper_final_20260618/pillow_same_color_cable_knit/dece_rf_flux/seed_10/result.png"),
 ("FLUX SD3-style gentle","outputs/flux_knitsd3_20260618/pillow_same_color_cable_knit/dece_rf_flux/seed_10/result.png"),
 ("FLUX broad-support","outputs/flux_knitbroad_20260618/pillow_same_color_cable_knit/dece_rf_flux/seed_10/result.png"),
]
cells=[]
for lbl,p in panels:
    im=z(p); cell=Image.new("RGB",(300,318),(255,255,255)); cell.paste(im,(0,18))
    ImageDraw.Draw(cell).text((3,4),lbl,fill=(0,0,0)); cells.append(cell)
W=300*2+8; H=318*2+8
c=Image.new("RGB",(W,H),(255,255,255))
c.paste(cells[0],(0,0)); c.paste(cells[1],(308,0)); c.paste(cells[2],(0,326)); c.paste(cells[3],(308,326))
c.save("/tmp/knit4.png"); print("saved",c.size)
