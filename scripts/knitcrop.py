from PIL import Image
import os
base="outputs/phase2_flux_paper_final_20260618"
for t,tag in [("pillow_same_color_cable_knit","white"),
              ("pillow_same_color_cable_knit_grey","grey"),
              ("pillow_same_color_cable_knit_armchair","arm")]:
    p=os.path.join(base,t,"dece_rf_flux","seed_10","result.png")
    im=Image.open(p).convert("RGB"); W,H=im.size
    im.save(f"/tmp/knit_{tag}_full.png")
    # zoom center where pillow sits (mask edge region) - crop middle 70%
    im.crop((int(W*0.18),int(H*0.18),int(W*0.82),int(H*0.82))).resize((460,460)).save(f"/tmp/knit_{tag}_zoom.png")
    print(tag, im.size)
