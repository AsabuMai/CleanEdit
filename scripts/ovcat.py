from PIL import Image
import os
B="data/pie_pilot_20260618/overlays"
ims=[]
for k in ["T3_2_01_C","T3_2_01_F","T5_7_00_C","T5_7_00_F"]:
    p=os.path.join(B,k+".png")
    if os.path.exists(p):
        im=Image.open(p).convert("RGB").resize((230,230)); ims.append((k,im))
from PIL import ImageDraw
W=sum(i[1].width for i in ims)+8*len(ims); H=250
c=Image.new("RGB",(W,H),(255,255,255)); x=0
for k,im in ims:
    c.paste(im,(x,18)); ImageDraw.Draw(c).text((x+2,4),k,fill=(0,0,0)); x+=im.width+8
c.save("/tmp/mask_orient.png"); print("ok",len(ims))
