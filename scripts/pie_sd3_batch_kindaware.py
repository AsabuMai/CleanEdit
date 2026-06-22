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
man=json.load(open(os.environ.get("MANIFEST",str(PROJ/"data/pie_pilot_20260618/manifest_pp_enriched.json"))))
LIMIT=int(os.environ.get("LIMIT",len(man))); OUT=PROJ/os.environ.get("OUT","outputs/pie_kindaware_sd3_20260618"); SEED=os.environ.get("SEED","10")
# SD3 per-task params.  T1-T3 are local edits on a subject; T4-T5 are
# subject-level attribute/material edits and need more subject-side freedom.
KIND={
      "t1_accessory":dict(operation="add_object",relation="inside_host",layering="none",hedit="1.10",text="0.12",rec="0.24",struct="0.50",core="1.00",subj="0.15",color=None),
      "t2_insert":dict(operation="add_object",relation="inside_container",layering="none",hedit="0.95",text="0.10",rec="0.26",struct="0.45",core="1.00",subj="0.25",color=None),
      "t3_text":dict(operation="add_decal",relation="inside_host",layering="none",hedit="1.05",text="0.18",rec="0.45",struct="0.45",core="1.05",subj="0.30",color=None),
      "t3_decal":dict(operation="add_decal",relation="on_surface",layering="none",hedit="0.85",text="0.10",rec="0.45",struct="0.45",core="0.95",subj="0.20",color=None),
      "t4_recolor":dict(operation="recolor",relation="inside_host",layering="recolor_trimap",hedit="0.22",text="0.02",rec="0.58",struct="0.50",core="0.85",subj="0.80",color="0.22"),
      "t5_material":dict(operation="replace",relation="inside_host",layering="none",hedit="0.86",text="0.14",rec="0.55",struct="0.70",core="1.00",subj="0.75",color=None),
      # Backward-compatible aliases for old overrides.
      "add":dict(operation="add_object",relation="inside_container",layering="none",hedit="0.95",text="0.10",rec="0.26",struct="0.45",core="1.00",subj="0.25",color=None),
      "decal":dict(operation="add_decal",relation="inside_host",layering="none",hedit="1.05",text="0.18",rec="0.45",struct="0.45",core="1.05",subj="0.30",color=None),
      "recolor":dict(operation="recolor",relation="inside_host",layering="recolor_trimap",hedit="0.22",text="0.02",rec="0.58",struct="0.50",core="0.85",subj="0.80",color="0.22"),
      "material":dict(operation="replace",relation="inside_host",layering="none",hedit="0.95",text="0.18",rec="0.38",struct="0.45",core="1.00",subj="0.75",color=None)}
FAM2KIND={"T1":"t1_accessory","F1":"t1_accessory",
          "T2":"t2_insert","F2":"t2_insert",
          "T3":"t3_decal","F3":"t3_decal",
          "T4":"t4_recolor","F4":"t4_recolor",
          "T5":"t5_material","F5":"t5_material"}
import os as _os, json as _json
_ov = _os.environ.get("KIND_OVERRIDE_JSON")
if _ov:
    for _k, _d in _json.loads(_ov).items():
        targets=[_k]
        if _k == "add": targets += ["t1_accessory","t2_insert"]
        elif _k == "decal": targets += ["t3_text","t3_decal"]
        elif _k == "recolor": targets += ["t4_recolor"]
        elif _k == "material": targets += ["t5_material"]
        for _target in targets:
            if _target in KIND:
                KIND[_target].update(_d)
    print("[kind-override]", _ov, flush=True)

COLORS=("black","white","gray","grey","red","orange","yellow","gold","golden","green","blue","cyan","teal","purple","violet","pink","brown","beige","silver")
def kind_of(e):
    cfg=e.get("config","")
    if cfg.startswith("2_attached_accessory"): return "t1_accessory"
    if cfg.startswith("2_container_insertion"): return "t2_insert"
    if cfg.startswith("6_"): return "t4_recolor"
    if cfg.startswith("7_"): return "t5_material"
    if cfg.startswith("3_"): return "t3_text" if is_text_surface_task(e) else "t3_decal"
    if cfg.startswith("2_"): return "t2_insert"
    family=e.get("family_label","") or e.get("family","")
    if family.startswith("T1") or family.startswith("F1"): return "t1_accessory"
    if family.startswith("T2") or family.startswith("F2"): return "t2_insert"
    if family.startswith("T3") or family.startswith("F3"): return "t3_text" if is_text_surface_task(e) else "t3_decal"
    if family.startswith("T4") or family.startswith("F4"): return "t4_recolor"
    if family.startswith("T5") or family.startswith("F5"): return "t5_material"
    return FAM2KIND.get(family[:2],"add")
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
def is_text_surface_task(e):
    s=" ".join(str(e.get(k,"")) for k in ("key","source_prompt","target_prompt","host_tokens","new_tokens")).lower()
    cues=("word","words","letter","letters","text","displays","sign","board","wifi","beer")
    return any(c in s for c in cues)
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
    amap=e.get("pp_aspect_mapping") or {}
    act=e.get("pp_edit_action") or {}
    if kind in ("add","decal","t1_accessory","t2_insert","t3_text","t3_decal"):
        host=list(amap.keys())
        new=[w for vals in amap.values() for w in vals] or list(act.keys())
    elif kind in ("recolor","material","t4_recolor","t5_material"):
        host=list(amap.keys())
        new=[w for vals in amap.values() for w in vals] or list(act.keys())
    else:
        new=list(act.keys()); host=list(amap.keys())
    return ",".join(new) if new else None, ",".join(host) if host else None

SUBJECT_PRESERVE_MODE=os.environ.get("SUBJECT_PRESERVE_MODE","0").lower() in {"1","true","yes","on"}
SUBJECT_TAR_GUIDANCE=os.environ.get("SUBJECT_TAR_GUIDANCE","8.0")
SUBJECT_CORE_SCALE=os.environ.get("SUBJECT_CORE_SCALE","0.90")
SUBJECT_EDIT_SCALE=os.environ.get("SUBJECT_EDIT_SCALE","0.20")
SUBJECT_TRAJ_SCALE=os.environ.get("SUBJECT_TRAJ_SCALE","0.50")
SUBJECT_SUPPORT_MAX_AREA=os.environ.get("SUBJECT_SUPPORT_MAX_AREA","0.25")
SUBJECT_OBJECT_PROVIDER=os.environ.get("SUBJECT_OBJECT_PROVIDER","operation_support_v3")
SUBJECT_GROUNDING_METHOD=os.environ.get(
    "SUBJECT_GROUNDING_METHOD",
    "external_mask" if SUBJECT_OBJECT_PROVIDER == "operation_support_v3" else "none",
)
SUBJECT_MASK_BLEND=os.environ.get("SUBJECT_MASK_BLEND","0").lower() in {"1","true","yes","on"}
SUBJECT_MASK_BLEND_MODE=os.environ.get("SUBJECT_MASK_BLEND_MODE","core")
T5_REF_ROOT=os.environ.get("T5_REF_ROOT")
T5_REF_SCALE=os.environ.get("T5_REF_SCALE","0.18")
T5_REF_CHROMA_MODE=os.environ.get("T5_REF_CHROMA_MODE","yuv_direction")
T5_REF_CHROMA_MAG=os.environ.get("T5_REF_CHROMA_MAG","0.65")
T5_REF_LUMA=os.environ.get("T5_REF_LUMA","0.75")
T5_REF_GRAD=os.environ.get("T5_REF_GRAD","0.35")
T5_REF_MAX_STRUCT_RATIO=os.environ.get("T5_REF_MAX_STRUCT_RATIO","0.45")
T5_POSE_PROMPT=os.environ.get("T5_POSE_PROMPT","1").lower() in {"1","true","yes","on"}
T5_PROMPT_PREFIX=os.environ.get("T5_PROMPT_PREFIX","").strip()
T5_NEGATIVE_PROMPT=os.environ.get("T5_NEGATIVE_PROMPT","").strip()
T5_EXPAND_EDIT_MASK=os.environ.get("T5_EXPAND_EDIT_MASK","1").lower() in {"1","true","yes","on"}
T5_EXPAND_EDIT_RADIUS=int(os.environ.get("T5_EXPAND_EDIT_RADIUS","15"))
T5_EXPAND_EDIT_MASK_BLUR=float(os.environ.get("T5_EXPAND_EDIT_MASK_BLUR","5.0"))
T5_EXPAND_RESTORE_BLUR=os.environ.get("T5_EXPAND_RESTORE_BLUR","3.0")

SUBJECT_KIND={
    "t1_accessory":dict(tar="8.0",core="1.00",subj="0.15",traj="0.45",max_area="0.28",min_area="0.05",relation="inside_host",layering="none",keep="1",dilate="7"),
    "t2_insert":dict(tar="8.0",core="1.00",subj="0.25",traj="0.45",max_area="0.35",min_area="0.06",relation="inside_container",layering="none",keep="3",dilate="5"),
    "t3_text":dict(tar="8.0",core="1.05",subj="0.30",traj="0.35",max_area="0.30",min_area="0.04",relation="inside_host",layering="none",keep="2",dilate="5"),
    "t3_decal":dict(tar="8.0",core="0.95",subj="0.20",traj="0.45",max_area="0.22",min_area="0.03",relation="on_surface",layering="none",keep="2",dilate="5"),
    "t4_recolor":dict(tar="7.0",core="0.85",subj="0.80",traj="0.25",max_area="0.35",min_area="0.08",relation="inside_host",layering="recolor_trimap",keep="2",dilate="5"),
    "t5_material":dict(tar="7.0",core="0.90",subj="0.50",traj="0.55",max_area="0.45",min_area="0.10",relation="inside_host",layering="none",keep="2",dilate="5",source_v="0.0",source_mask="edit",source_steps="12"),
}
_sov=_os.environ.get("SUBJECT_KIND_OVERRIDE_JSON")
if _sov:
    for _k,_d in _json.loads(_sov).items():
        targets=[_k]
        if _k == "add": targets += ["t1_accessory","t2_insert"]
        elif _k == "decal": targets += ["t3_text","t3_decal"]
        elif _k == "recolor": targets += ["t4_recolor"]
        elif _k == "material": targets += ["t5_material"]
        for _target in targets:
            SUBJECT_KIND.setdefault(_target,{}).update(_d)
    print("[subject-kind-override]", _sov, flush=True)

def subject_params(kind):
    p=dict(SUBJECT_KIND.get(kind,{}))
    if "SUBJECT_TAR_GUIDANCE" in os.environ: p["tar"]=SUBJECT_TAR_GUIDANCE
    if "SUBJECT_CORE_SCALE" in os.environ: p["core"]=SUBJECT_CORE_SCALE
    if "SUBJECT_EDIT_SCALE" in os.environ: p["subj"]=SUBJECT_EDIT_SCALE
    if "SUBJECT_TRAJ_SCALE" in os.environ: p["traj"]=SUBJECT_TRAJ_SCALE
    if "SUBJECT_SUPPORT_MAX_AREA" in os.environ: p["max_area"]=SUBJECT_SUPPORT_MAX_AREA
    return p

def set_arg(argv, flag, value):
    out=[]; i=0
    while i < len(argv):
        if argv[i] == flag:
            i += 2
        else:
            out.append(argv[i]); i += 1
    out += [flag, str(value)]
    argv[:] = out

def expanded_t5_mask(mask, entry, radius=T5_EXPAND_EDIT_RADIUS, blur=T5_EXPAND_EDIT_MASK_BLUR):
    if not mask or radius <= 0:
        return mask
    from PIL import Image, ImageFilter
    src=Path(mask)
    out_dir=PROJ/"data/flowedit_compatible_135/derived_masks"
    out_dir.mkdir(parents=True,exist_ok=True)
    safe_key=re.sub(r"[^A-Za-z0-9_.-]+","_",entry.get("key",src.stem))
    out=out_dir/f"{safe_key}_{src.stem}_expand_r{radius}_b{blur:g}.png"
    if not out.exists():
        im=Image.open(src).convert("L")
        im=im.filter(ImageFilter.MaxFilter(radius*2+1))
        if blur > 0:
            im=im.filter(ImageFilter.GaussianBlur(blur))
        im.save(out)
        print("[t5-expand-mask]",src,"->",out,flush=True)
    return str(out)

done=0
for e in man[:LIMIT]:
    k=kind_of(e); P=KIND[k]; od=OUT/e["key"]/"support_v3_controller_rmsgap"/f"seed_{SEED}"
    if (od/"result.png").exists(): print("skip",e["key"]); continue
    od.mkdir(parents=True,exist_ok=True)
    target_prompt=e["target_prompt"]
    if SUBJECT_PRESERVE_MODE and k == "t5_material" and T5_PROMPT_PREFIX:
        prefix=T5_PROMPT_PREFIX.rstrip(".")
        if prefix.lower() not in target_prompt.lower():
            target_prompt=prefix+". "+target_prompt
    if SUBJECT_PRESERVE_MODE and k == "t5_material" and T5_POSE_PROMPT:
        suffix="Preserve the exact original pose, silhouette, camera view, and background; only the subject material and surface appearance change."
        if suffix.lower() not in target_prompt.lower():
            target_prompt=target_prompt.rstrip(".")+". "+suffix
    argv=["--image",str((PROJ/e["image"]).resolve()),"--source-prompt",e["source_prompt"],"--prompt",target_prompt,
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
    if SUBJECT_PRESERVE_MODE and k == "t5_material" and T5_NEGATIVE_PROMPT:
        argv+=["--negative-prompt",T5_NEGATIVE_PROMPT]
    if SUBJECT_PRESERVE_MODE:
        SP=subject_params(k)
        set_arg(argv,"--tar-guidance-scale",SP.get("tar",SUBJECT_TAR_GUIDANCE))
        set_arg(argv,"--edit-core-scale",SP.get("core",SUBJECT_CORE_SCALE))
        set_arg(argv,"--edit-subject-scale",SP.get("subj",SUBJECT_EDIT_SCALE))
        set_arg(argv,"--trajectory-subject-preserve-scale",SP.get("traj",SUBJECT_TRAJ_SCALE))
        set_arg(argv,"--object-mask-provider",SUBJECT_OBJECT_PROVIDER)
        set_arg(argv,"--grounding-method",SUBJECT_GROUNDING_METHOD)
        set_arg(argv,"--relation",SP.get("relation",P["relation"]))
        set_arg(argv,"--mask-layering-mode",SP.get("layering","none"))
        set_arg(argv,"--support-max-area-ratio",SP.get("max_area",SUBJECT_SUPPORT_MAX_AREA))
        if SP.get("min_area"): set_arg(argv,"--support-min-area-ratio",SP["min_area"])
        if SP.get("keep"): set_arg(argv,"--support-keep-components",SP["keep"])
        if SP.get("dilate"): set_arg(argv,"--support-dilate-radius",SP["dilate"])
        if SP.get("source_v"): set_arg(argv,"--source-inject-v-scale",SP["source_v"])
        if SP.get("source_mask"): set_arg(argv,"--source-inject-mask-mode",SP["source_mask"])
        if SP.get("source_steps"): set_arg(argv,"--source-inject-steps",SP["source_steps"])
        if SUBJECT_MASK_BLEND:
            argv += ["--mask-blend","--mask-blend-mode",SUBJECT_MASK_BLEND_MODE]
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
        final_mask=mask
        final_mode="subject_core" if SUBJECT_PRESERVE_MODE else "replace"
        if SUBJECT_PRESERVE_MODE and k == "t5_material" and T5_EXPAND_EDIT_MASK:
            final_mask=expanded_t5_mask(mask,e)
            final_mode="replace"
        argv+=["--support-mask",mask,"--final-edit-mask",final_mask,
               "--final-edit-mask-mode",final_mode,
               "--final-outside-restore-mask",final_mask,"--final-outside-restore-scale","1.0"]
        if SUBJECT_PRESERVE_MODE and k == "t5_material" and T5_EXPAND_EDIT_MASK:
            argv+=["--final-outside-restore-mask-blur",T5_EXPAND_RESTORE_BLUR]
        if not SUBJECT_PRESERVE_MODE:
            argv+=["--object-mask-provider","semantic"]
    if SUBJECT_PRESERVE_MODE and k == "t5_material" and T5_REF_ROOT:
        ref_path=PROJ/T5_REF_ROOT/e["key"]/"support_v3_controller_rmsgap"/f"seed_{SEED}"/"result.png"
        if ref_path.exists():
            argv+=["--edit-ref-guidance-scale",T5_REF_SCALE,
                   "--edit-ref-image",str(ref_path),
                   "--edit-ref-structure-image",str((PROJ/e["image"]).resolve()),
                   "--edit-ref-chroma-mode",T5_REF_CHROMA_MODE,
                   "--edit-ref-chroma-magnitude-scale",T5_REF_CHROMA_MAG,
                   "--edit-ref-luma-preserve-scale",T5_REF_LUMA,
                   "--edit-ref-gradient-preserve-scale",T5_REF_GRAD,
                   "--edit-ref-max-struct-rms-ratio",T5_REF_MAX_STRUCT_RATIO]
            if mask: argv+=["--edit-ref-mask",mask]
        else:
            print("[t5-ref-missing]",ref_path,flush=True)
    _adapt = __import__("os").environ.get("ADAPT_OVERRIDE_JSON")
    if _adapt:
        for _fl, _vl in json.loads(_adapt).get(k, {}).items():
            set_arg(argv, _fl, _vl)
    sys.argv=["run_edit_sd3.py"]+argv; ts=time.time()
    try: run_edit_sd3.main(); done+=1; print("OK",e["key"],k,"%.1fs"%(time.time()-ts),flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FAILED",e["key"],repr(ex),flush=True)
print("ALLDONE kindaware-sd3",done,"/",min(LIMIT,len(man)),flush=True)
