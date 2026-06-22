import os, sys, json, re, types, time
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project"); sys.path.insert(0, str(PROJ))
import torch, run_edit_sd3
_orig=run_edit_sd3.StableDiffusion3Pipeline.from_pretrained; _cache={}
def _fp(*a,**k):
    key=str(a[0]) if a else "d"
    if key not in _cache:
        p=_orig(*a,**k)
        p.enable_model_cpu_offload=types.MethodType(lambda self,*x,**y:None,p)
        p.enable_sequential_cpu_offload=types.MethodType(lambda self,*x,**y:None,p)
        p.enable_attention_slicing=types.MethodType(lambda self,*x,**y:None,p)
        p.to("cuda"); _cache[key]=p; print("[ka-sd3] pipe loaded once",flush=True)
    return _cache[key]
run_edit_sd3.StableDiffusion3Pipeline.from_pretrained=staticmethod(_fp)
man=json.load(open(os.environ.get("MANIFEST",str(PROJ/"data/flowedit_compatible_123/manifest.json"))))
LIMIT=int(os.environ.get("LIMIT",len(man))); OUT=PROJ/os.environ.get("OUT","outputs/flowedit_compatible_sd3_20260619"); SEED=os.environ.get("SEED","10")
# SD3 per-kind params (from pretty_matrix command.txt)
KIND={"add":dict(operation="add_object",relation="on_surface",layering="object_contact",hedit="0.65",text="0.08",rec="0.22",struct="0.45",core="1.35",subj="0.35",color=None),
      "decal":dict(operation="add_decal",relation="on_surface",layering="object_contact",hedit="0.65",text="0.08",rec="0.45",struct="0.45",core="1.35",subj="0.35",color=None),
      "material":dict(operation="replace",relation="none",layering="none",hedit="0.68",text="0.14",rec="0.40",struct="0.45",core="1.35",subj="0.35",color=None),
      "recolor":dict(operation="recolor",relation="none",layering="recolor_trimap",hedit="0.18",text="0.02",rec="0.58",struct="0.45",core="1.35",subj="1.0",color="0.10")}
KIND["accessory"]=dict(KIND["add"])
KIND["insert"]=dict(KIND["add"])
KIND["accessory"].update(hedit="0.60",text="0.08",rec="0.32",struct="0.50",subj="0.35")
KIND["insert"].update(hedit="0.78",text="0.12",rec="0.18",struct="0.40",subj="0.42")
KIND["add"].update(hedit="0.74",text="0.11",rec="0.18",struct="0.40",subj="0.42")
KIND["decal"].update(hedit="0.78",text="0.16",rec="0.36",struct="0.40",subj="0.50")
KIND["material"].update(hedit="0.74",text="0.16",rec="0.42",struct="0.46",subj="0.35")
KIND["recolor"].update(hedit="0.22",text="0.03",rec="0.54",struct="0.43",subj="1.0",color="0.12")
FAM2KIND={"T1":"accessory","T2":"insert","T3":"decal","T4":"recolor","T5":"material",
          "F1":"accessory","F2":"insert","F3":"decal","F4":"recolor","F5":"material"}
COLORS=("black","white","gray","grey","red","orange","yellow","gold","golden","green","blue","cyan","teal","purple","violet","pink","brown","beige","silver")
def kind_of(e):
    fam=e.get("family","")[:2]
    if fam in FAM2KIND: return FAM2KIND[fam]
    cfg=e.get("config","")
    if cfg.startswith("6_"): return "recolor"
    if cfg.startswith("7_"): return "material"
    if cfg.startswith("3_"): return "decal"
    if cfg.startswith("2_"): return "add"
    return FAM2KIND.get(e.get("family","T1")[:2],"add")
def _colors(s):
    s=s.lower().replace("-"," ")
    return [c for c in COLORS if re.search(r"\b"+re.escape(c)+r"\b",s)]
def recolor_colors(e):
    act=e.get("pp_edit_action") or {}
    for word,meta in act.items():
        if meta.get("edit_type")==6 and word.lower() in COLORS:
            src=meta.get("action")
            return (src.lower() if isinstance(src,str) and src!="+" else None), word.lower()
    src=_colors(e.get("source_prompt",""))
    tgt=_colors(e.get("target_prompt",""))
    source=src[0] if src else None
    target=None
    for c in tgt:
        if c not in set(src):
            target=c; break
    target=target or (tgt[-1] if tgt else None)
    return source,target
def local_mask(e):
    pm=e.get("pp_local_mask")
    if pm:
        p=(PROJ/pm).resolve()
        return str(p) if p.exists() else None
    if e.get("global_mask", True):
        return None
    m=e.get("mask")
    p=(PROJ/m).resolve() if m else None
    return str(p) if p and p.exists() else None
def pp_tokens(e, kind):
    if e.get("new_tokens") or e.get("host_tokens"):
        return e.get("new_tokens") or None, e.get("host_tokens") or None
    amap=e.get("pp_aspect_mapping") or {}
    act=e.get("pp_edit_action") or {}
    if kind=="add":
        new=list(amap.keys()) or list(act.keys())
        host=[]
    elif kind in ("recolor","material"):
        host=list(amap.keys())
        new=[w for vals in amap.values() for w in vals] or list(act.keys())
    else:
        new=list(act.keys()); host=list(amap.keys())
    return ",".join(new) if new else None, ",".join(host) if host else None
done=0
for e in man[:LIMIT]:
    k=kind_of(e); P=KIND[k]; od=OUT/e["key"]/"support_v3_controller_rmsgap"/f"seed_{SEED}"
    if (od/"result.png").exists(): print("skip",e["key"]); continue
    od.mkdir(parents=True,exist_ok=True)
    argv=["--image",str((PROJ/e["image"]).resolve()),"--source-prompt",e["source_prompt"],"--prompt",e["target_prompt"],
      "--output",str(od/"result.png"),"--stats-output",str(od/"stats.json"),"--metadata-output",str(od/"metadata.json"),"--mask-output-dir",str(od/"masks"),
      "--max-image-size","512","--seed",SEED,"--num-inference-steps","28","--n-max","24",
      "--src-guidance-scale","1.0","--base-guidance-scale","1.0","--tar-guidance-scale","10.5",
      "--edit-hedit-guidance-scale",P["hedit"],"--edit-guidance-scale","0.0","--edit-region-guidance-scale","0.0",
      "--edit-target-guidance-scale","0.0","--edit-source-guidance-scale","0.0",
      "--edit-text-guidance-scale",P["text"],"--edit-text-source-scale","0.8","--edit-text-core-weight","1.0","--edit-text-subject-weight","0.3",
      "--rec-guidance-scale",P["rec"],"--struct-guidance-scale",P["struct"],"--trajectory-preserve-scale","0.25","--trajectory-subject-preserve-scale","0.0",
      "--edit-core-scale",P["core"],"--edit-subject-scale",P["subj"],"--region-target-transport-scale","0.0","--region-target-outside-lock-scale","0.0",
      "--rec-stop-timestep","0.08","--beta-max","1.0","--velocity-conversion-mode","linear_path","--linear-path-t-min","0.05",
      "--object-mask-provider","attention_velocity","--grounding-method","none","--edit-operation",P["operation"],"--relation",P["relation"],"--mask-layering-mode",P["layering"],
      "--adaptive-clean-control","--adaptive-edit-target-rms","0.42","--adaptive-rmsgap-mode","legacy","--adaptive-preserve-drift-budget","0.12",
      "--adaptive-edit-gain","2.0","--adaptive-preserve-gain","4.2","--adaptive-edit-weight-min","0.85","--adaptive-edit-weight-max","1.55",
      "--adaptive-preserve-weight-min","1.0","--adaptive-preserve-weight-max","1.65","--adaptive-projection-scale","0.65","--adaptive-preserve-clean-correction-scale","0.5",
      "--photo-prompt-mode","both","--log-every","7"]
    new_tokens,host_tokens=pp_tokens(e,k)
    if new_tokens: argv+=["--new-tokens",new_tokens]
    if host_tokens: argv+=["--host-tokens",host_tokens]
    if P["color"] is not None:
        source,target=recolor_colors(e)
        argv+=["--edit-color-guidance-scale",P["color"],"--edit-color-clean-projection-scale","0.18","--edit-color-texture-preserve-scale","0.10"]
        if source: argv+=["--edit-color-source",source]
        if target: argv+=["--edit-color-target",target]
    mask=local_mask(e)
    if mask:
        argv+=["--support-mask",mask,"--object-mask-provider","semantic","--final-edit-mask",mask,"--final-edit-mask-mode","replace",
               "--final-outside-restore-mask",mask,"--final-outside-restore-scale","1.0"]
    sys.argv=["run_edit_sd3.py"]+argv; ts=time.time()
    try: run_edit_sd3.main(); done+=1; print("OK",e["key"],k,"%.1fs"%(time.time()-ts),flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FAILED",e["key"],repr(ex),flush=True)
print("ALLDONE kindaware-sd3",done,"/",min(LIMIT,len(man)),flush=True)
