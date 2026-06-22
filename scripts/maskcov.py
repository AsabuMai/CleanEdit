from PIL import Image
import numpy as np, os
base="outputs/phase2_flux_paper_final_20260618"
tasks=["pillow_same_color_cable_knit","pillow_same_color_cable_knit_grey","pillow_same_color_cable_knit_armchair"]
for t in tasks:
    md=os.path.join(base,t,"dece_rf_flux","seed_10","masks")
    row=[t]
    for f in ["subject_final.png","core_final.png","preserve_final.png"]:
        p=os.path.join(md,f)
        if os.path.exists(p):
            m=np.asarray(Image.open(p).convert("L"))
            row.append("%s=%.3f"%(f.split('_')[0], (m>30).mean()))
        else:
            row.append(f+"=NA")
    print("  ".join(row))
# also save a side-by-side of subject vs core for white and grey
for t,tag in [("pillow_same_color_cable_knit","white"),("pillow_same_color_cable_knit_grey","grey")]:
    md=os.path.join(base,t,"dece_rf_flux","seed_10","masks")
    sub=Image.open(os.path.join(md,"subject_final.png")).convert("L").resize((200,200))
    core=Image.open(os.path.join(md,"core_final.png")).convert("L").resize((200,200))
    c=Image.new("L",(410,200),0); c.paste(sub,(0,0)); c.paste(core,(210,0))
    c.save(f"/tmp/mask_{tag}.png")
print("saved mask vis")
