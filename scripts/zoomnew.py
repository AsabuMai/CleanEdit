from PIL import Image
import os
new="outputs/flux_knitfill_20260618"
for t,tag in [("pillow_same_color_cable_knit","white"),("pillow_same_color_cable_knit_armchair","arm")]:
    im=Image.open(os.path.join(new,t,"dece_rf_flux","seed_10","result.png")).convert("RGB"); W,H=im.size
    im.crop((int(W*0.18),int(H*0.18),int(W*0.82),int(H*0.82))).resize((460,460)).save(f"/tmp/knitnew_{tag}.png")
print("ok")
