import json, os, urllib.request, ssl, time
import numpy as np
from PIL import Image
ctx=ssl.create_default_context()
BASE="data/pie_dev_20260618"; os.makedirs(f"{BASE}/images",exist_ok=True); os.makedirs(f"{BASE}/masks",exist_ok=True)
# pick localized-mask entries per category (caps); these give real bg-preservation signal
PLAN={"1_change_object_80":2,"2_add_object_80":2,"3_delete_object_80":4,"6_change_attribute_color_40":2,"7_change_attribute_material_40":4}
def fetch(cfg):
    url=f"https://datasets-server.huggingface.co/rows?dataset=UB-CVML-Group/PIE_Bench_pp&config={cfg}&split=V1&offset=0&length=100"
    for _ in range(4):
        try: return json.load(urllib.request.urlopen(url,timeout=40,context=ctx))["rows"]
        except Exception as ex: print("retry",ex); time.sleep(3)
    raise RuntimeError(cfg)
def dec(s):
    n=list(map(int,s.split())); f=np.zeros(512*512,np.uint8)
    for i in range(0,len(n)-1,2): f[n[i]:n[i]+n[i+1]]=255
    return f.reshape(512,512)
def strip(p): return p.replace("[","").replace("]","")
man=[]
for cfg,cap in PLAN.items():
    rows=fetch(cfg); taken=0
    for row in rows:
        if taken>=cap: break
        r=row["row"]; mk=r["mask"].strip()
        if mk=="0 262144": continue   # localized only
        m=dec(mk); frac=m.mean()/255
        if frac<0.02 or frac>0.6: continue  # sane localized size
        key=f"{cfg.split('_')[0]}_{row['row_idx']:02d}"
        ip=f"{BASE}/images/{key}.jpg"
        for _ in range(4):
            try: urllib.request.urlretrieve(r["image"]["src"],ip); break
            except Exception as ex: print("imgretry",ex); time.sleep(3)
        Image.fromarray(m).save(f"{BASE}/masks/{key}.png")
        man.append({"key":key,"config":cfg,"row_idx":row["row_idx"],"id":r["id"],
            "image":ip,"mask":f"{BASE}/masks/{key}.png","global_mask":False,
            "source_prompt":strip(r["source_prompt"]),"target_prompt":strip(r["target_prompt"]),
            "mask_frac":round(frac,3)})
        taken+=1
        print(f"{key:10s} frac={frac:.3f} | {r['source_prompt'][:40]} -> {r['target_prompt'][:44]}")
json.dump(man,open(f"{BASE}/manifest.json","w"),indent=2)
print("WROTE",len(man),"entries ->",f"{BASE}/manifest.json")
