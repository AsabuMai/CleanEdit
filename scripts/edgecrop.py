from PIL import Image
import numpy as np, os
base="outputs/phase2_flux_paper_final_20260618"
def dump(t, name):
    p=os.path.join(base,t,"dece_rf_flux","seed_10","result.png")
    im=Image.open(p).convert("RGB"); W,H=im.size
    print(name, "size", im.size)
    return im,W,H
# MUG: salmon bg. crop right edge band + bottom-left handle/base
mug,W,H=dump("green_mug_orange_phase2","MUG")
mug.crop((int(W*0.45),int(H*0.55),int(W*0.95),int(H*0.92))).resize((400,300)).save("/tmp/crop_mug.png")
# VASE: white cloth bg. crop bottom + left edge of vase base
v,W,H=dump("yellow_vase_blue_phase2","VASE")
v.crop((int(W*0.05),int(H*0.62),int(W*0.95),int(H*0.95))).resize((480,180)).save("/tmp/crop_vase.png")
print("ok")
