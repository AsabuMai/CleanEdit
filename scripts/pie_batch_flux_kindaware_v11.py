import os, sys, json, re, time
from pathlib import Path
import torch
_PROJ_PATH=Path("/cluster/users/grad/2025/25t8103/project")
sys.path.insert(0, str(_PROJ_PATH))
sys.path.insert(0, str(_PROJ_PATH/"flux"))
import flux_hrec
from flux_hrec import build_parser, load_flux_pipeline, HRecFluxEdit
PROJ=_PROJ_PATH
man=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/pie_pilot_20260618/manifest_pp_enriched.json"))))
START=int(os.environ.get("START","0"))
LIMIT=int(os.environ.get("LIMIT", len(man)))
OUT=PROJ/os.environ.get("FLUX_OUT","outputs/pie_kindaware_flux_20260618"); SEED=os.environ.get("SEED","10")
CACHE_DIR=os.environ.get("FLUX_CACHE_DIR", str(PROJ/".cache/huggingface/hub"))
FLUX_STEPS=os.environ.get("FLUX_STEPS","12")
FLUX_N_MAX=os.environ.get("FLUX_N_MAX","10")
FLUX_MAX_SEQ=os.environ.get("FLUX_MAX_SEQ","512")
FLUX_LOCAL_FILES_ONLY=os.environ.get("FLUX_LOCAL_FILES_ONLY","1").lower() in {"1","true","yes","on"}
FLUX_OFFLOAD=os.environ.get("FLUX_OFFLOAD","model").lower()
FLUX_PROMPT_ENCODE_DEVICE=os.environ.get("FLUX_PROMPT_ENCODE_DEVICE","cuda").lower()
FLUX_KEEP_TEXT_ENCODERS_CPU=os.environ.get("FLUX_KEEP_TEXT_ENCODERS_CPU","0").lower() in {"1","true","yes","on"}
T3_SPELL_TEXT=os.environ.get("T3_SPELL_TEXT","0").lower() in {"1","true","yes","on"}
T3_SPELL_MAX_LEN=int(os.environ.get("T3_SPELL_MAX_LEN","6"))
T3_SURFACE_PROMPT=os.environ.get("T3_SURFACE_PROMPT","0").lower() in {"1","true","yes","on"}
T3_COMPACT_PROMPT=os.environ.get("T3_COMPACT_PROMPT","0").lower() in {"1","true","yes","on"}
T5_POSE_PROMPT=os.environ.get("T5_POSE_PROMPT","1").lower() in {"1","true","yes","on"}
T5_PROMPT_PREFIX=os.environ.get("T5_PROMPT_PREFIX","").strip()
T5_AUTO_PROMPT_PREFIX=os.environ.get("T5_AUTO_PROMPT_PREFIX","1").lower() in {"1","true","yes","on"}
T5_NEGATIVE_PROMPT=os.environ.get("T5_NEGATIVE_PROMPT","").strip()
T5_EXPAND_EDIT_MASK=os.environ.get("T5_EXPAND_EDIT_MASK","1").lower() in {"1","true","yes","on"}
T5_EXPAND_EDIT_RADIUS=int(os.environ.get("T5_EXPAND_EDIT_RADIUS","21"))
T5_EXPAND_EDIT_MASK_BLUR=float(os.environ.get("T5_EXPAND_EDIT_MASK_BLUR","6.0"))
KIND={
 "t1_accessory":dict(operation="add_object",relation="on_surface",layering="object_contact",edit_hedit=0.95,edit_anchor=0.18,edit_region=0.32,edit_target=0.18,edit_source=0.02,local_target=0.95,region_transport=0.35,core=82.0,outside_lock=0.03,rec=0.18,struct=0.40,traj=0.12,pbud=0.20,pgain=2.2),
 "t2_insert":dict(operation="add_object",relation="on_surface",layering="object_contact",edit_hedit=0.95,edit_anchor=0.18,edit_region=0.32,edit_target=0.18,edit_source=0.02,local_target=0.95,region_transport=0.35,core=82.0,outside_lock=0.03,rec=0.18,struct=0.40,traj=0.12,pbud=0.20,pgain=2.2),
 "t3_text":dict(operation="add_decal",relation="on_surface",layering="object_contact",edit_hedit=0.98,edit_anchor=0.18,edit_region=0.34,edit_target=0.18,edit_source=0.015,local_target=0.90,region_transport=0.24,core=86.0,outside_lock=0.0,rec=0.28,struct=0.40,traj=0.18,pbud=0.14,pgain=3.2),
 "t3_decal":dict(operation="add_decal",relation="on_surface",layering="object_contact",edit_hedit=0.98,edit_anchor=0.18,edit_region=0.34,edit_target=0.18,edit_source=0.015,local_target=0.90,region_transport=0.24,core=86.0,outside_lock=0.0,rec=0.28,struct=0.40,traj=0.18,pbud=0.14,pgain=3.2),
 "t4_recolor":dict(operation="recolor",relation="none",layering="none",edit_hedit=0.55,edit_anchor=0.06,edit_region=0.14,edit_target=0.04,edit_source=0.0,local_target=0.18,region_transport=0.00,core=None,outside_lock=0.00,rec=0.50,struct=0.45,traj=0.40,pbud=0.10,pgain=3.0),
 "t5_material":dict(operation="replace",relation="none",layering="none",edit_hedit=1.30,edit_anchor=0.14,edit_region=0.30,edit_target=0.26,edit_source=0.01,local_target=1.05,region_transport=0.00,core=70.0,outside_lock=0.02,rec=0.15,struct=0.40,traj=0.10,pbud=0.12,pgain=1.6),
 # Backward-compatible aliases for old smoke scripts and overrides.
 "add":dict(operation="add_object",relation="on_surface",layering="object_contact",edit_hedit=0.95,edit_anchor=0.18,edit_region=0.32,edit_target=0.18,edit_source=0.02,local_target=0.95,region_transport=0.35,core=82.0,outside_lock=0.03,rec=0.18,struct=0.40,traj=0.12,pbud=0.20,pgain=2.2),
 "decal":dict(operation="add_decal",relation="on_surface",layering="object_contact",edit_hedit=0.98,edit_anchor=0.18,edit_region=0.34,edit_target=0.18,edit_source=0.015,local_target=0.90,region_transport=0.24,core=86.0,outside_lock=0.0,rec=0.28,struct=0.40,traj=0.18,pbud=0.14,pgain=3.2),
 "recolor":dict(operation="recolor",relation="none",layering="none",edit_hedit=0.55,edit_anchor=0.06,edit_region=0.14,edit_target=0.04,edit_source=0.0,local_target=0.18,region_transport=0.00,core=None,outside_lock=0.00,rec=0.50,struct=0.45,traj=0.40,pbud=0.10,pgain=3.0),
 "material":dict(operation="replace",relation="none",layering="none",edit_hedit=1.30,edit_anchor=0.14,edit_region=0.30,edit_target=0.26,edit_source=0.01,local_target=1.05,region_transport=0.00,core=70.0,outside_lock=0.02,rec=0.15,struct=0.40,traj=0.10,pbud=0.12,pgain=1.6)}
FAM2KIND={"T1":"t1_accessory","F1":"t1_accessory",
          "T2":"t2_insert","F2":"t2_insert",
          "T3":"t3_decal","F3":"t3_decal",
          "T4":"t4_recolor","F4":"t4_recolor",
          "T5":"t5_material","F5":"t5_material"}
import os as _os, json as _json
def override_targets(k):
    targets=[k]
    if k == "add": targets += ["t1_accessory","t2_insert"]
    elif k == "decal": targets += ["t3_text","t3_decal"]
    elif k == "recolor": targets += ["t4_recolor"]
    elif k == "material": targets += ["t5_material"]
    return targets
_ov = _os.environ.get("KIND_OVERRIDE_JSON")
if _ov:
    for _k, _d in _json.loads(_ov).items():
        for _target in override_targets(_k):
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
    return FAM2KIND.get(family[:2],"t1_accessory")
def _colors(s):
    s=s.lower().replace("-"," ")
    return [c for c in COLORS if re.search(r"\b"+re.escape(c)+r"\b",s)]
def recolor_target(e):
    act=e.get("pp_edit_action") or {}
    for word,meta in act.items():
        if meta.get("edit_type")==6 and word.lower() in COLORS:
            return word.lower()
    src=set(_colors(e.get("source_prompt","")))
    tgt=_colors(e.get("target_prompt",""))
    for c in tgt:
        if c not in src:
            return c
    return tgt[-1] if tgt else None
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
def final_blend_mask(e):
    fm=e.get("final_blend_mask")
    if not fm:
        return None
    p=(PROJ/fm).resolve()
    return str(p) if p.exists() else None
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
    out=out_dir/f"{safe_key}_{src.stem}_flux_expand_r{radius}_b{blur:g}.png"
    if not out.exists() or src.stat().st_mtime > out.stat().st_mtime:
        im=Image.open(src).convert("L")
        im=im.filter(ImageFilter.MaxFilter(radius*2+1))
        if blur > 0:
            im=im.filter(ImageFilter.GaussianBlur(blur))
        im.save(out)
        print("[flux-t5-expand-mask]",src,"->",out,flush=True)
    return str(out)
def pp_tokens(e, kind):
    if e.get("new_tokens") or e.get("host_tokens"):
        return e.get("new_tokens") or None, e.get("host_tokens") or None
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
def t3_spelling_suffix(new_tokens):
    if not T3_SPELL_TEXT or not new_tokens:
        return ""
    parts=[]
    for tok in str(new_tokens).split(","):
        clean=re.sub(r"[^A-Za-z0-9]+","",tok).strip()
        if clean.isalpha() and 2 <= len(clean) <= T3_SPELL_MAX_LEN:
            up=clean.upper()
            parts.append(f'The target surface text must read exactly "{up}", with the letters {" ".join(up)}.')
    return " ".join(parts)
def t3_surface_suffix(entry, new_tokens, host_tokens):
    if not T3_SURFACE_PROMPT or not new_tokens:
        return ""
    key=str(entry.get("key","")).lower()
    word=str(new_tokens).split(",")[0].strip().upper()
    host=(str(host_tokens).split(",")[0].strip() if host_tokens else "surface")
    base=(
        f'Preserve the exact original {host} shape, pose, perspective, lighting, material, and background. '
        f'Replace only the existing surface text with "{word}". '
        'The new text is physically printed, painted, handwritten, or made from the same neon/sign material on the original surface, not a sticker, not an overlay, and not a rectangular patch.'
    )
    if "stop_arrow" in key or re.search(r"_stop_[123]_", key):
        return base+f' Keep the same red octagonal sign and pole; only the word on the red sign changes to "{word}".'
    if "gas_station" in key:
        return base+f' Keep the same white rectangular gas station sign panel and canopy; only the red letters on the panel change to "{word}".'
    if "groceries" in key:
        return base+f' Keep the same paper color, hand, clip, and list layout; replace only the first handwritten list item with "- {word}".'
    if "_sign_" in key:
        return base+f' Keep the same billboard geometry and black typography style; replace only the first word with "{word}" while preserving "IS ALL YOU NEED".'
    if "luna" in key:
        return base+f' Keep the same red neon sign box; render "{word}" as neon tubing on the sign.'
    if "this_must_be_the_place" in key:
        return base+f' Keep the same glass/acrylic sign and glowing white tubing; replace only the top word with "{word}".'
    return base
def t3_compact_prompt(entry, new_tokens, host_tokens):
    if not T3_COMPACT_PROMPT or not new_tokens:
        return ""
    key=str(entry.get("key","")).lower()
    word=str(new_tokens).split(",")[0].strip().upper()
    if "stop_arrow" in key or re.search(r"_stop_[123]_", key):
        return f'Same red octagonal stop sign, pole, arrow, lighting, and background; replace only the word on the sign with "{word}". Printed on the sign surface, not sticker or overlay.'
    if "gas_station" in key:
        return f'Same gas station sign panel, canopy, perspective, lighting, and background; replace only the red sign letters with "{word}". Printed on the sign surface, not sticker or overlay.'
    if "groceries" in key:
        return f'Same paper grocery list in the hand with the same clip, layout, lighting, and background; replace only the first handwritten item with "- {word}". Ink on paper, not sticker or overlay.'
    if "_sign_" in key:
        return f'Same billboard geometry, perspective, black typography style, and background; replace only the first word with "{word}" and keep "IS ALL YOU NEED". Printed on the billboard, not sticker or overlay.'
    if "luna" in key:
        return f'Same red neon sign box, tubing glow, wall, perspective, lighting, and background; change only the neon word to "{word}". Neon tubing on the sign, not sticker or overlay.'
    if "this_must_be_the_place" in key:
        return f'Same glass acrylic glowing sign, tubing, reflections, perspective, and background; replace only the top glowing word with "{word}". Built into the sign, not sticker or overlay.'
    host=(str(host_tokens).split(",")[0].strip() if host_tokens else "surface")
    return f'Same {host}, perspective, lighting, material, and background; replace only the existing surface text with "{word}". On the original surface, not sticker or overlay.'
def _phrase(csv):
    if not csv:
        return ""
    words=[w.strip().replace("_"," ") for w in str(csv).split(",") if w.strip()]
    return " ".join(words)
def auto_t5_prefix(e):
    new_tokens, host_tokens = pp_tokens(e, "t5_material")
    material = _phrase(new_tokens)
    host = _phrase(host_tokens) or "subject"
    if not material:
        return ""
    return f"A clean-edged {material} version of the {host}, coherent shape"
def argv_for(e, od, P):
    kind=kind_of(e)
    t=e["target_prompt"]
    new_tokens,host_tokens=pp_tokens(e, kind)
    if kind in ("t3_text","t3_decal"):
        compact=t3_compact_prompt(e,new_tokens,host_tokens)
        if compact:
            t=compact
    if kind == "t5_material":
        prefix=T5_PROMPT_PREFIX.rstrip(".")
        if not prefix and T5_AUTO_PROMPT_PREFIX:
            prefix=auto_t5_prefix(e).rstrip(".")
        if prefix.lower() not in t.lower():
            t=prefix+". "+t
    if kind == "t5_material" and T5_POSE_PROMPT:
        suffix="Preserve pose, silhouette, camera, and background; only material changes."
        if suffix.lower() not in t.lower():
            t=t.rstrip(".")+". "+suffix
    if kind in ("t3_text","t3_decal"):
        suffix=t3_surface_suffix(e,new_tokens,host_tokens)
        if suffix and suffix.lower() not in t.lower():
            t=t.rstrip(".")+". "+suffix
        suffix=t3_spelling_suffix(new_tokens)
        if suffix and suffix.lower() not in t.lower():
            t=t.rstrip(".")+". "+suffix
    a=["--image",str((PROJ/e["image"]).resolve()),"--source-prompt",e["source_prompt"],"--prompt",t,
      "--output",str(od/"result.png"),"--metadata-output",str(od/"metadata.json"),"--stats-output",str(od/"stats.json"),"--mask-output-dir",str(od/"masks"),
      "--method","dece_rf_flux","--seed",SEED,"--num-inference-steps",FLUX_STEPS,"--n-max",FLUX_N_MAX,"--max-image-size","512","--max-sequence-length",FLUX_MAX_SEQ,
      "--src-guidance-scale","1.0","--base-guidance-scale","1.0","--tar-guidance-scale","5.0",
      "--support-control-mode","operation","--use-flux-attention-support",
      "--edit-operation",P["operation"],"--support-relation",P["relation"],"--mask-layering-mode",P["layering"],
      "--edit-hedit-guidance-scale",str(P["edit_hedit"]),"--edit-guidance-scale",str(P["edit_anchor"]),
      "--edit-region-guidance-scale",str(P["edit_region"]),"--edit-target-guidance-scale",str(P["edit_target"]),"--edit-source-guidance-scale",str(P["edit_source"]),
      "--edit-local-target-prompt",t,"--edit-local-target-guidance-scale",str(P["local_target"]),"--edit-local-target-cfg-scale","5.0",
      "--rec-guidance-scale",str(P["rec"]),"--struct-guidance-scale",str(P["struct"]),"--trajectory-preserve-scale",str(P["traj"]),
      "--beta-max","1.0","--rec-stop-timestep","0.08","--linear-path-t-min","0.05",
      "--adaptive-clean-control","--adaptive-edit-target-rms","0.42","--adaptive-rmsgap-mode","legacy",
      "--adaptive-preserve-drift-budget",str(P["pbud"]),"--adaptive-edit-gain","2.0","--adaptive-preserve-gain",str(P["pgain"]),
      "--adaptive-edit-weight-min","0.85","--adaptive-edit-weight-max","1.55","--adaptive-preserve-weight-min","1.0","--adaptive-preserve-weight-max","1.65",
      "--adaptive-projection-scale","0.65","--adaptive-preserve-clean-correction-scale","0.5",
      "--region-target-transport-scale",str(P["region_transport"]),"--region-target-outside-lock-scale",str(P["outside_lock"]),
      "--final-postprocess-mode","mask_blend","--final-mask-blend-scale","1.0","--final-mask-alpha-gamma","1.0",
      "--cache-dir",CACHE_DIR,
      "--true-cfg","--distilled-guidance","1.0"]
    if FLUX_LOCAL_FILES_ONLY:
        a+=["--local-files-only"]
    if FLUX_OFFLOAD == "sequential":
        a+=["--sequential-offload"]
    elif FLUX_OFFLOAD == "model":
        a+=["--model-offload"]
    if FLUX_KEEP_TEXT_ENCODERS_CPU:
        a+=["--keep-text-encoders-cpu"]
    if FLUX_PROMPT_ENCODE_DEVICE:
        a+=["--prompt-encode-device",FLUX_PROMPT_ENCODE_DEVICE]
    if kind == "t5_material" and T5_NEGATIVE_PROMPT:
        a+=["--negative-prompt",T5_NEGATIVE_PROMPT]
    if P["core"] is not None: a+=["--fixed-core-from-attention","--fixed-core-attention-percentile",str(P["core"])]
    if new_tokens: a+=["--new-tokens",new_tokens]
    if host_tokens: a+=["--host-tokens",host_tokens]
    if P["operation"]=="recolor":
        target=recolor_target(e)
        if target: a+=["--recolor-target",target]
        import os as _o
        a+=["--minimal-recolor-scale",_o.environ.get("RC_MINIMAL","0.0"),
            "--final-recolor-blend-scale",_o.environ.get("RC_BLEND","0.0"),
            "--recolor-clean-projection-scale",_o.environ.get("RC_PROJ","0.0"),
            "--recolor-texture-preserve-scale",_o.environ.get("RC_TEXPRES","0.85"),
            "--recolor-preserve-luma-scale",_o.environ.get("RC_LUMA","0.65"),
            "--recolor-clean-projection-alpha",_o.environ.get("RC_ALPHA","0.85")]
    mask=local_mask(e)
    _mov=__import__("os").environ.get("MASK_OVERRIDE_JSON")
    if _mov:
        _md=json.loads(_mov); _ek=str(e.get("key","")).lower()
        for _pat,_mp in _md.items():
            if _pat.lower() in _ek:
                mask=_mp
    if kind == "t5_material" and T5_EXPAND_EDIT_MASK:
        mask=expanded_t5_mask(mask,e)
    if mask:
        a+=["--semantic-base-mask",mask,"--support-control-mode","fixed","--support-external-mask-role","attention"]
    fmask=final_blend_mask(e)
    if fmask:
        a+=["--final-blend-mask",fmask]
    _adapt = __import__("os").environ.get("ADAPT_OVERRIDE_JSON")
    if _adapt:
        adapt=json.loads(_adapt)
        alias={"t1_accessory":"add","t2_insert":"add","t3_text":"decal","t3_decal":"decal","t4_recolor":"recolor","t5_material":"material"}.get(kind)
        bucket={}
        if alias:
            bucket.update(adapt.get(alias, {}))
        bucket.update(adapt.get(kind, {}))
        for _fl, _vl in bucket.items():
            set_arg(a,_fl,_vl)
    _keyov = __import__("os").environ.get("KEY_OVERRIDE_JSON")
    if _keyov:
        _ko=json.loads(_keyov)
        _ek=str(e.get("key","")).lower()
        for _pat,_m in _ko.items():
            if _pat.lower() in _ek:
                for _fl,_vl in _m.items():
                    set_arg(a,_fl,_vl)
    _abl=__import__("os").environ.get("ABLATE","")
    if _abl=="preserve":
        for _fl in ("--rec-guidance-scale","--struct-guidance-scale","--trajectory-preserve-scale","--adaptive-preserve-drift-budget","--adaptive-preserve-gain","--adaptive-projection-scale","--adaptive-preserve-clean-correction-scale","--region-target-outside-lock-scale"):
            set_arg(a,_fl,"0")
    elif _abl=="adaptive":
        if "--adaptive-clean-control" in a: a.remove("--adaptive-clean-control")
    elif _abl=="opsupport":
        set_arg(a,"--support-control-mode","fixed")
        if "--use-flux-attention-support" in a: a.remove("--use-flux-attention-support")
    elif _abl=="no_final_postprocess":
        set_arg(a,"--final-postprocess-mode","none")
        set_arg(a,"--final-mask-blend-scale","0")
        set_arg(a,"--final-recolor-blend-scale","0")
    return a
work=man[START:START+LIMIT]
if not work:
    raise SystemExit(f"empty manifest slice START={START} LIMIT={LIMIT} total={len(man)}")
dev=torch.device("cuda"); 
e0=work[0]; od0=OUT/e0["key"]/"dece_rf_flux"/f"seed_{SEED}"; od0.mkdir(parents=True,exist_ok=True)
base=build_parser().parse_args(argv_for(e0,od0,KIND[kind_of(e0)]))
t0=time.time(); PIPE=load_flux_pipeline(base,dev); print("[ka-flux] loaded once %.1fs"%(time.time()-t0),flush=True)
flux_hrec.load_flux_pipeline=lambda a,d=dev: PIPE
done=0
for e in work:
    k=kind_of(e); P=KIND[k]; od=OUT/e["key"]/"dece_rf_flux"/f"seed_{SEED}"; od.mkdir(parents=True,exist_ok=True)
    if (od/"result.png").exists():
        print("skip",e["key"],flush=True); continue
    args=build_parser().parse_args(argv_for(e,od,P)); ts=time.time()
    try:
        res=HRecFluxEdit(args); res.images[0].save(args.output)
        json.dump(res.metadata,open(args.metadata_output,"w")); json.dump(res.stats,open(args.stats_output,"w"))
        done+=1; print("OK",e["key"],k,"%.1fs"%(time.time()-ts),flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FAILED",e["key"],repr(ex),flush=True)
print("ALLDONE kindaware-flux",done,"/",len(work),flush=True)
