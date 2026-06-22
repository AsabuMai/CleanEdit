from PIL import Image
import os
base="outputs/pretty_matrix"; m="support_v3_controller_rmsgap"
for t,tag in [("pillow_same_color_cable_knit_grey","grey"),("pillow_same_color_cable_knit","white"),("pillow_same_color_cable_knit_armchair","arm")]:
    p=os.path.join(base,t,m,"seed_10","result.png")
    im=Image.open(p).convert("RGB"); W,H=im.size
    im.crop((int(W*0.18),int(H*0.18),int(W*0.82),int(H*0.82))).resize((440,440)).save(f"/tmp/sd3z_{tag}.png")
print("ok")
