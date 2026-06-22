from PIL import Image
import numpy as np, os
base="outputs/phase2_flux_paper_final_20260618/dog_bow_tie_phase2/dece_rf_flux/seed_10"
im=Image.open(os.path.join(base,"result.png")).convert("RGB"); W,H=im.size
print("img",im.size)
im.crop((int(W*0.25),int(H*0.40),int(W*0.78),int(H*0.70))).resize((460,260)).save("/tmp/bow_zoom.png")
# mask coverage + bbox
for f in ["subject_final.png","core_final.png"]:
    p=os.path.join(base,"masks",f)
    if os.path.exists(p):
        m=np.asarray(Image.open(p).convert("L"))>30
        ys,xs=np.where(m)
        if len(xs):
            print(f,"frac=%.3f bbox x[%d-%d]/%d y[%d-%d]/%d"%(m.mean(),xs.min(),xs.max(),m.shape[1],ys.min(),ys.max(),m.shape[0]))
        Image.fromarray((m*255).astype('uint8')).resize((200,200)).save(f"/tmp/bowmask_{f.split('_')[0]}.png")
