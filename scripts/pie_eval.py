import json, csv, sys, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from collections import defaultdict
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
man=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/pie_pilot_20260618/manifest.json"))))
if os.environ.get("METHODS_SPEC"):
    METHODS={}
    for part in os.environ["METHODS_SPEC"].split(";"):
        lab,d,sub=part.split(":"); METHODS[lab]=(d,sub)
else:
    METHODS={"flux":(os.environ.get("FLUX_OUT","outputs/pie_auto_flux_20260618"),"dece_rf_flux"),
             "sd3":(os.environ.get("SD3_OUT","outputs/pie_auto_sd3_20260618"),"support_v3_controller_rmsgap")}
SEED="10"; dev="cuda" if torch.cuda.is_available() else "cpu"
# models
from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModel
import lpips as lpipslib
from skimage.metrics import structural_similarity
clipm=CLIPModel.from_pretrained("openai/clip-vit-large-patch14",local_files_only=True).to(dev).eval()
clipp=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",local_files_only=True)
dinop=AutoImageProcessor.from_pretrained("facebook/dinov2-base",local_files_only=True)
dinom=AutoModel.from_pretrained("facebook/dinov2-base",local_files_only=True).to(dev).eval()
lp=lpipslib.LPIPS(net="alex").to(dev).eval()
def load(p,sz=512): return np.asarray(Image.open(p).convert("RGB").resize((sz,sz)),np.float32)/255.0
def mask_decode_png(p):
    m=(np.asarray(Image.open(p).convert("L").resize((512,512)))>127).astype(np.float32)
    m[0,:]=1;m[-1,:]=1;m[:,0]=1;m[:,-1]=1
    return m
def clip_img(arr):
    im=Image.fromarray((arr*255).astype(np.uint8))
    with torch.no_grad():
        x=clipp(images=im,return_tensors="pt").to(dev)
        f=clipm.get_image_features(**x); return torch.nn.functional.normalize(f,dim=-1)
def clip_txt(t):
    with torch.no_grad():
        x=clipp(text=[t],return_tensors="pt",padding=True,truncation=True).to(dev)
        f=clipm.get_text_features(**x); return torch.nn.functional.normalize(f,dim=-1)
def dino_feat(arr):
    im=Image.fromarray((arr*255).astype(np.uint8))
    with torch.no_grad():
        x=dinop(images=im,return_tensors="pt").to(dev)
        o=dinom(**x); f=o.last_hidden_state[:,0]; return torch.nn.functional.normalize(f,dim=-1)
def lpips_bg(src,res,bg):
    a=(src*bg[...,None]); b=(res*bg[...,None])
    ta=torch.from_numpy(a.transpose(2,0,1)[None]*2-1).float().to(dev)
    tb=torch.from_numpy(b.transpose(2,0,1)[None]*2-1).float().to(dev)
    with torch.no_grad(): return float(lp(ta,tb).item())
rows=[]
for e in man:
    src=load(e["image"]); m=mask_decode_png(e["mask"]); bg=1.0-m
    tgt=e["target_prompt"]; tfeat=clip_txt(tgt)
    bg_px=bg.astype(bool); has_bg=bg_px.sum()>50
    for meth,(d,sub) in METHODS.items():
        rp=PROJ/d/e["key"]/sub/f"seed_{SEED}"/"result.png"
        if not rp.exists(): print("MISS",meth,e["key"]); continue
        res=load(str(rp))
        diff=(src-res)**2
        mse_bg=float(diff[bg_px].mean()) if has_bg else float("nan")
        psnr_bg=10*np.log10(1.0/max(mse_bg,1e-10)) if has_bg else float("nan")
        ya=(0.299*src[...,0]+0.587*src[...,1]+0.114*src[...,2])
        yb=(0.299*res[...,0]+0.587*res[...,1]+0.114*res[...,2])
        _,smap=structural_similarity(ya,yb,data_range=1.0,full=True)
        ssim_bg=float(smap[bg_px].mean()) if has_bg else float("nan")
        lp_bg=lpips_bg(src,res,bg) if has_bg else float("nan")
        cw=float((clip_img(res)@tfeat.T).item())
        ys,xs=np.where(m>0.5); 
        if len(xs): crop=res[ys.min():ys.max()+1, xs.min():xs.max()+1]
        else: crop=res
        ce=float((clip_img(crop)@tfeat.T).item())
        dino=float((dino_feat(src)@dino_feat(res).T).item())
        rows.append(dict(key=e["key"],family=e.get("family", e.get("config","?")),method=meth,global_mask=int(e["global_mask"]),
            bg_psnr=psnr_bg,bg_mse=mse_bg,bg_ssim=ssim_bg,bg_lpips=lp_bg,
            clip_whole=cw,clip_edit=ce,dino_struct=dino))
        print(meth,e["key"],"clipW %.3f clipE %.3f dino %.3f bgPSNR %.1f bgLPIPS %.3f"%(cw,ce,dino,psnr_bg,lp_bg)); sys.stdout.flush()
outd=PROJ/os.environ.get("EVAL_OUT","experiments/pie_pilot_eval_20260618"); outd.mkdir(parents=True,exist_ok=True)
with open(outd/"metrics.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# summary per method (overall + localized-only for bg)
def mean(xs): xs=[x for x in xs if x==x]; return sum(xs)/len(xs) if xs else float("nan")
print("\n===== SUMMARY (mean) =====")
for meth in METHODS:
    R=[r for r in rows if r["method"]==meth]
    Rl=[r for r in R if r["global_mask"]==0]
    print(f"[{meth}] n={len(R)} clip_whole={mean([r['clip_whole'] for r in R]):.4f} clip_edit={mean([r['clip_edit'] for r in R]):.4f} dino_struct={mean([r['dino_struct'] for r in R]):.4f}")
    print(f"      preservation on LOCALIZED-mask only (n={len(Rl)}): bg_psnr={mean([r['bg_psnr'] for r in Rl]):.2f} bg_lpips={mean([r['bg_lpips'] for r in Rl]):.4f} bg_ssim={mean([r['bg_ssim'] for r in Rl]):.4f}")
print("WROTE", outd/"metrics.csv")
