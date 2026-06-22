from PIL import Image
import numpy as np, os
d="outputs/phase2_dece_flux_custom_masks_20260616"
for f in ["dog_bow_tie_neck.png","dog_bow_tie_wide.png"]:
    p=os.path.join(d,f)
    if os.path.exists(p):
        m=np.asarray(Image.open(p).convert("L"))>30
        ys,xs=np.where(m)
        print(f,"size",m.shape,"frac=%.3f"%m.mean(),"x[%d-%d](%.2f-%.2f) y[%d-%d](%.2f-%.2f)"%(xs.min(),xs.max(),xs.min()/m.shape[1],xs.max()/m.shape[1],ys.min(),ys.max(),ys.min()/m.shape[0],ys.max()/m.shape[0]))
        Image.open(p).convert("L").resize((180,180)).save(f"/tmp/mc_{f.replace('.png','')}.png")
    else:
        print(f,"MISSING")
