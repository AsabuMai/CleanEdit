import json, os, urllib.request, ssl, time
import numpy as np
from PIL import Image
ctx = ssl.create_default_context()
BASE="data/pie_pilot_20260618"
os.makedirs(f"{BASE}/images", exist_ok=True)
os.makedirs(f"{BASE}/masks", exist_ok=True)
os.makedirs(f"{BASE}/overlays", exist_ok=True)
# (family, config, row_idx)
SEL=[
 ("T1","2_add_object_80",18),("T1","2_add_object_80",40),("T1","2_add_object_80",42),
 ("T2","2_add_object_80",68),("T2","2_add_object_80",65),("T2","2_add_object_80",64),
 ("T3","2_add_object_80",1),("T3","2_add_object_80",11),("T3","2_add_object_80",23),
 ("T4","6_change_attribute_color_40",14),("T4","6_change_attribute_color_40",30),("T4","6_change_attribute_color_40",18),
 ("T5","7_change_attribute_material_40",0),("T5","7_change_attribute_material_40",13),("T5","7_change_attribute_material_40",30),
]
def fetch_rows(cfg):
    url=f"https://datasets-server.huggingface.co/rows?dataset=UB-CVML-Group/PIE_Bench_pp&config={cfg}&split=V1&offset=0&length=100"
    for _ in range(4):
        try:
            return json.load(urllib.request.urlopen(url, timeout=40, context=ctx))["rows"]
        except Exception as e:
            print("retry",cfg,e); time.sleep(3)
    raise RuntimeError("fetch failed "+cfg)
def decode_mask(s):
    n=list(map(int,s.split())); flat=np.zeros(512*512,np.uint8)
    for i in range(0,len(n)-1,2):
        st=n[i]; ln=n[i+1]; flat[st:st+ln]=255
    return flat
def strip(p): return p.replace("[","").replace("]","")
cache={}
manifest=[]
for fam,cfg,idx in SEL:
    if cfg not in cache: cache[cfg]=fetch_rows(cfg)
    r=cache[cfg][idx]["row"]
    key=f"{fam}_{cfg.split('_')[0]}_{idx:02d}"
    # image
    img_url=r["image"]["src"]
    ip=f"{BASE}/images/{key}.jpg"
    for _ in range(4):
        try: urllib.request.urlretrieve(img_url, ip); break
        except Exception as e: print("img retry",e); time.sleep(3)
    im=Image.open(ip).convert("RGB"); W,H=im.size
    # mask (decode at 512, then resize to image if needed). PIE images are 512.
    flat=decode_mask(r["mask"])
    glob = (r["mask"].strip()=="0 262144") or (flat.mean()>0.999*255)
    mc=flat.reshape(512,512)            # C order
    mp=f"{BASE}/masks/{key}.png"
    Image.fromarray(mc).resize((W,H)).save(mp)
    # overlay for the localized one(s)
    if not glob:
        mf=flat.reshape(512,512,order='F')
        ov=im.resize((512,512)).copy(); arr=np.asarray(ov).astype(np.float32)
        red=np.zeros_like(arr); red[...,0]=255
        for tag,mm in [("C",mc),("F",mf)]:
            a=(mm[...,None]/255.0)*0.5
            blend=(arr*(1-a)+red*a).astype(np.uint8)
            Image.fromarray(blend).save(f"{BASE}/overlays/{key}_{tag}.png")
    manifest.append({"key":key,"family":fam,"config":cfg,"row_idx":idx,"id":r["id"],
        "image":ip,"mask":mp,"global_mask":bool(glob),
        "source_prompt":strip(r["source_prompt"]),"target_prompt":strip(r["target_prompt"])})
    print(f"{key:18s} glob={int(glob)} maskfrac={mc.mean()/255:.3f} | {r['source_prompt'][:40]} -> {r['target_prompt'][:48]}")
json.dump(manifest, open(f"{BASE}/manifest.json","w"), indent=2)
print("WROTE", f"{BASE}/manifest.json", "n=",len(manifest))
