import os, sys, json, re, time, hashlib
from pathlib import Path
import torch
_PROJ_PATH=Path(os.environ.get("CLEANEDIT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(_PROJ_PATH))
sys.path.insert(0, str(_PROJ_PATH/"flux"))
import flux_hrec
from flux_hrec import build_parser, load_flux_pipeline, HRecFluxEdit
PROJ=_PROJ_PATH
_expected_root=os.environ.get("EXPECTED_PROJECT_ROOT")
if _expected_root and _PROJ_PATH != Path(_expected_root).resolve():
    raise SystemExit(f"project root mismatch: {_PROJ_PATH} != {Path(_expected_root).resolve()}")

man=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/flowedit_compatible_135/manifest.json"))))
START=int(os.environ.get("START","0"))
LIMIT=int(os.environ.get("LIMIT", len(man)))
OUT=PROJ/os.environ.get("FLUX_OUT","outputs/flowedit135_flux"); SEED=os.environ.get("SEED","10")
CACHE_DIR=os.environ.get("FLUX_CACHE_DIR", str(PROJ/".cache/huggingface/hub"))
FLUX_STEPS=os.environ.get("FLUX_STEPS","12")
FLUX_N_MAX=os.environ.get("FLUX_N_MAX","10")
FLUX_MAX_SEQ=os.environ.get("FLUX_MAX_SEQ","512")
FLUX_LOCAL_FILES_ONLY=os.environ.get("FLUX_LOCAL_FILES_ONLY","1").lower() in {"1","true","yes","on"}
FLUX_OFFLOAD=os.environ.get("FLUX_OFFLOAD","model").lower()
FLUX_PROMPT_ENCODE_DEVICE=os.environ.get("FLUX_PROMPT_ENCODE_DEVICE","cuda").lower()
RUN_PURPOSE=os.environ.get("RUN_PURPOSE","diagnostic").lower()
ADAPTIVE_COMPONENT_CONTROL=os.environ.get("ADAPTIVE_COMPONENT_CONTROL","0").lower() in {"1","true","yes","on"}
FLUX_KEEP_TEXT_ENCODERS_CPU=os.environ.get("FLUX_KEEP_TEXT_ENCODERS_CPU","0").lower() in {"1","true","yes","on"}
T3_SPELL_TEXT=os.environ.get("T3_SPELL_TEXT","0").lower() in {"1","true","yes","on"}
T3_SPELL_MAX_LEN=int(os.environ.get("T3_SPELL_MAX_LEN","6"))
T3_SURFACE_PROMPT=os.environ.get("T3_SURFACE_PROMPT","0").lower() in {"1","true","yes","on"}
T3_COMPACT_PROMPT=os.environ.get("T3_COMPACT_PROMPT","0").lower() in {"1","true","yes","on"}
T5_POSE_PROMPT=os.environ.get("T5_POSE_PROMPT","1").lower() in {"1","true","yes","on"}
T5_PROMPT_PREFIX=os.environ.get("T5_PROMPT_PREFIX","").strip()
T5_AUTO_PROMPT_PREFIX=os.environ.get("T5_AUTO_PROMPT_PREFIX","1").lower() in {"1","true","yes","on"}
T5_NEGATIVE_PROMPT=os.environ.get("T5_NEGATIVE_PROMPT","").strip()
T3_NEGATIVE_PROMPT=os.environ.get("T3_NEGATIVE_PROMPT","").strip()
T3_NEG_FROM_REMOVED=os.environ.get("T3_NEG_FROM_REMOVED","0").lower() in {"1","true","yes","on"}
T3_TWO_STAGE=os.environ.get("T3_TWO_STAGE","0").lower() in {"1","true","yes","on"}
T3_ERASE_FILL_SCALE=float(os.environ.get("T3_ERASE_FILL_SCALE","1.0"))
T3_ERASE_SUPPRESSION_SCALE=float(os.environ.get("T3_ERASE_SUPPRESSION_SCALE","0.55"))
T3_ERASE_RING_REC_SCALE=float(os.environ.get("T3_ERASE_RING_REC_SCALE","0.35"))
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
def t3_removed_words(entry):
    src_w=re.findall(r"[A-Za-z]+", entry["source_prompt"].lower())
    tgt_w=set(re.findall(r"[A-Za-z]+", entry["target_prompt"].lower()))
    stop={"a","an","the","of","with","and","that","reads","says","word","words","text","sign","above","below","on","in","there","are","is"}
    return [w.upper() for w in dict.fromkeys(src_w) if w not in tgt_w and w not in stop and len(w)>=3][:3]
def t3_blank_prompt(entry):
    key=str(entry.get("key","")).lower()
    if "gas_station" in key:
        return "The same gas station, cars, canopy, lighting, and background, with a clean blank white and red sign panel containing no letters or text."
    if "groceries" in key:
        return "The same grocery list on the same brown paper, with the first handwritten item cleanly removed while EGGS and MILK, the magnet, lighting, hand, and background remain unchanged."
    if "stop_arrow" in key:
        return "The same red octagonal sign, pole, blue arrow sign, lighting, and background, with the red sign face clean and blank, containing no letters or text."
    host=(str(entry.get("host_tokens") or "surface").split(",")[0].strip())
    return f"The same image with the existing writing cleanly removed from the {host}, leaving a natural blank {host} with no letters or text; everything else remains unchanged."
def t3_negative_for_words(words):
    if not words:
        return "letters, words, text, ghost letters, double exposure"
    old=", ".join('the word "%s"'%word for word in words)
    return old+", old text showing through, ghost letters, double exposure"
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
    t3_erase_stage=bool(e.get("_t3_erase_stage"))
    if kind in ("t3_text","t3_decal") and not t3_erase_stage:
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
    if kind in ("t3_text","t3_decal") and not t3_erase_stage:
        suffix=t3_surface_suffix(e,new_tokens,host_tokens)
        if suffix and suffix.lower() not in t.lower():
            t=t.rstrip(".")+". "+suffix
        suffix=t3_spelling_suffix(new_tokens)
        if suffix and suffix.lower() not in t.lower():
            t=t.rstrip(".")+". "+suffix
    a=["--image",str((PROJ/e["image"]).resolve()),"--source-prompt",e["source_prompt"],"--prompt",t,
      "--output",str(od/"result.png"),"--metadata-output",str(od/"metadata.json"),"--stats-output",str(od/"stats.json"),"--mask-output-dir",str(od/"masks"),
      "--method","dece_rf_flux","--run-purpose",RUN_PURPOSE,"--comparison-protocol",("diagnostic_multi_pass" if T3_TWO_STAGE else "single_pass"),
      "--seed",SEED,"--num-inference-steps",FLUX_STEPS,"--n-max",FLUX_N_MAX,"--max-image-size","512","--max-sequence-length",FLUX_MAX_SEQ,
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
    if ADAPTIVE_COMPONENT_CONTROL:
        a+=["--adaptive-component-control"]
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
    if kind in ("t3_text","t3_decal") and (T3_NEGATIVE_PROMPT or T3_NEG_FROM_REMOVED):
        neg=T3_NEGATIVE_PROMPT
        if T3_NEG_FROM_REMOVED:
            olds=t3_removed_words(e)
            if olds:
                neg=(neg+", " if neg else "")+t3_negative_for_words(olds)
                print("[t3-neg]",e["key"],"->",neg,flush=True)
        if neg:
            a+=["--negative-prompt",neg]
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
recipe="dece_rf_flux_two_stage" if T3_TWO_STAGE else "dece_rf_flux"
e0=work[0]; od0=OUT/e0["key"]/recipe/f"seed_{SEED}"; od0.mkdir(parents=True,exist_ok=True)
base=build_parser().parse_args(argv_for(e0,od0,KIND[kind_of(e0)]))
t0=time.time(); PIPE=load_flux_pipeline(base,dev); print("[ka-flux] loaded once %.1fs"%(time.time()-t0),flush=True)
flux_hrec.load_flux_pipeline=lambda a,d=dev: PIPE
def env_json(name):
    raw=os.environ.get(name,"").strip()
    return json.loads(raw) if raw else {}
def canonical_sha256(value):
    payload=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
def batch_recipe_config():
    return {
        "ablate":os.environ.get("ABLATE",""),
        "run_purpose":RUN_PURPOSE,
        "adaptive_component_control":ADAPTIVE_COMPONENT_CONTROL,
        "t3_neg_from_removed":T3_NEG_FROM_REMOVED,
        "t3_compact_prompt":T3_COMPACT_PROMPT,
        "t3_spell_text":T3_SPELL_TEXT,
        "t3_surface_prompt":T3_SURFACE_PROMPT,
        "t5_pose_prompt":T5_POSE_PROMPT,
        "t5_auto_prompt_prefix":T5_AUTO_PROMPT_PREFIX,
        "t5_prompt_prefix":T5_PROMPT_PREFIX,
        "t5_negative_prompt":T5_NEGATIVE_PROMPT,
        "t5_expand_edit_mask":T5_EXPAND_EDIT_MASK,
        "t5_expand_edit_radius":T5_EXPAND_EDIT_RADIUS,
        "t5_expand_edit_mask_blur":T5_EXPAND_EDIT_MASK_BLUR,
        "kind_override":env_json("KIND_OVERRIDE_JSON"),
        "key_override":env_json("KEY_OVERRIDE_JSON"),
        "mask_override":env_json("MASK_OVERRIDE_JSON"),
        "adapt_override":env_json("ADAPT_OVERRIDE_JSON"),
    }
def save_result(res,args):
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    res.images[0].save(args.output)
    recipe_config=batch_recipe_config()
    res.metadata["batch_recipe_config"]=recipe_config
    res.metadata["batch_recipe_config_sha256"]=canonical_sha256(recipe_config)
    json.dump(res.metadata,open(args.metadata_output,"w"),indent=2)
    json.dump(res.stats,open(args.stats_output,"w"),indent=2)
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()
done=0
for e in work:
    k=kind_of(e); P=KIND[k]; od=OUT/e["key"]/recipe/f"seed_{SEED}"; od.mkdir(parents=True,exist_ok=True)
    if (od/"result.png").exists():
        print("skip",e["key"],flush=True); continue
    ts=time.time()
    try:
        if T3_TWO_STAGE and k in ("t3_text","t3_decal"):
            old_words=t3_removed_words(e)
            blank_prompt=t3_blank_prompt(e)
            erase_entry=dict(e)
            erase_entry["target_prompt"]=blank_prompt
            erase_entry["new_tokens"]="blank"
            erase_entry["_t3_erase_stage"]=True
            erase_params=dict(P,operation="remove_object",relation="none",layering="none",core=None,edit_source=0.0)
            erase_dir=od/"stage1_erase"; erase_dir.mkdir(parents=True,exist_ok=True)
            erase_argv=argv_for(erase_entry,erase_dir,erase_params)
            set_arg(erase_argv,"--edit-operation","remove_object")
            set_arg(erase_argv,"--removal-controller-mode","clean_fill")
            set_arg(erase_argv,"--removal-fill-scale",T3_ERASE_FILL_SCALE)
            set_arg(erase_argv,"--removal-suppression-scale",T3_ERASE_SUPPRESSION_SCALE)
            set_arg(erase_argv,"--removal-ring-rec-scale",T3_ERASE_RING_REC_SCALE)
            set_arg(erase_argv,"--negative-prompt",t3_negative_for_words(old_words))
            erase_args=build_parser().parse_args(erase_argv)
            erase_res=HRecFluxEdit(erase_args)
            erase_res.metadata["two_stage_role"]="erase_fill"
            erase_res.metadata["two_stage_blank_prompt"]=blank_prompt
            erase_res.metadata["two_stage_removed_words"]=old_words
            save_result(erase_res,erase_args)

            write_entry=dict(e)
            write_entry["image"]=str(erase_dir/"result.png")
            write_entry["source_prompt"]=blank_prompt
            write_dir=od/"stage2_write"; write_dir.mkdir(parents=True,exist_ok=True)
            write_argv=argv_for(write_entry,write_dir,P)
            set_arg(write_argv,"--negative-prompt",t3_negative_for_words(old_words))
            write_args=build_parser().parse_args(write_argv)
            write_res=HRecFluxEdit(write_args)
            write_res.metadata["two_stage_role"]="write"
            write_res.metadata["two_stage_input"]=str(erase_dir/"result.png")
            write_res.metadata["two_stage_input_sha256"]=sha256(erase_dir/"result.png")
            write_res.metadata["two_stage_original_source_prompt"]=e["source_prompt"]
            save_result(write_res,write_args)
            write_res.images[0].save(od/"result.png")
            json.dump({
                "enabled":True,
                "erase_output":str(erase_dir/"result.png"),
                "erase_sha256":sha256(erase_dir/"result.png"),
                "write_output":str(write_dir/"result.png"),
                "write_sha256":sha256(write_dir/"result.png"),
                "final_output":str(od/"result.png"),
                "final_sha256":sha256(od/"result.png"),
                "blank_prompt":blank_prompt,
                "removed_words":old_words,
            },open(od/"two_stage.json","w"),indent=2)
        else:
            args=build_parser().parse_args(argv_for(e,od,P))
            res=HRecFluxEdit(args)
            save_result(res,args)
        done+=1; print("OK",e["key"],k,"%.1fs"%(time.time()-ts),flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FAILED",e["key"],repr(ex),flush=True)
print("ALLDONE kindaware-flux",done,"/",len(work),flush=True)
