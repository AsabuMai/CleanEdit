import os, sys, json, re, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path("/cluster/users/grad/2025/25t8103/project")/"flux"))
import flux_hrec
from flux_hrec import build_parser, load_flux_pipeline, HRecFluxEdit
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
man=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/pie_pilot_20260618/manifest_pp_enriched.json"))))
LIMIT=int(os.environ.get("LIMIT", len(man)))
OUT=PROJ/os.environ.get("FLUX_OUT","outputs/pie_kindaware_flux_20260618"); SEED="10"
KIND={
 "add":dict(operation="add_object",relation="on_surface",layering="object_contact",edit_hedit=0.95,edit_anchor=0.18,edit_region=0.32,edit_target=0.18,edit_source=0.02,local_target=0.95,region_transport=0.35,core=82.0,outside_lock=0.03,rec=0.18,struct=0.40,traj=0.12,pbud=0.20,pgain=2.2),
 "decal":dict(operation="add_decal",relation="on_surface",layering="object_contact",edit_hedit=0.98,edit_anchor=0.18,edit_region=0.34,edit_target=0.18,edit_source=0.015,local_target=0.90,region_transport=0.24,core=86.0,outside_lock=0.0,rec=0.28,struct=0.40,traj=0.18,pbud=0.14,pgain=3.2),
 "material":dict(operation="replace",relation="none",layering="none",edit_hedit=1.30,edit_anchor=0.14,edit_region=0.30,edit_target=0.26,edit_source=0.01,local_target=1.05,region_transport=0.00,core=70.0,outside_lock=0.02,rec=0.15,struct=0.40,traj=0.10,pbud=0.12,pgain=1.6),
 "recolor":dict(operation="recolor",relation="none",layering="none",edit_hedit=0.36,edit_anchor=0.06,edit_region=0.14,edit_target=0.04,edit_source=0.0,local_target=0.18,region_transport=0.00,core=None,outside_lock=0.00,rec=0.50,struct=0.45,traj=0.40,pbud=0.10,pgain=3.0)}
FAM2KIND={"T1":"add","T2":"add","T3":"decal","T4":"recolor","T5":"material"}
import os as _os, json as _json
_ov = _os.environ.get("KIND_OVERRIDE_JSON")
if _ov:
    for _k, _d in _json.loads(_ov).items():
        if _k in KIND:
            KIND[_k].update(_d)
    print("[kind-override]", _ov, flush=True)

COLORS=("black","white","gray","grey","red","orange","yellow","gold","golden","green","blue","cyan","teal","purple","violet","pink","brown","beige","silver")
def kind_of(e):
    cfg=e.get("config","")
    if cfg.startswith("6_"): return "recolor"
    if cfg.startswith("7_"): return "material"
    if cfg.startswith("3_"): return "decal"
    if cfg.startswith("2_"): return "add"
    return FAM2KIND.get(e.get("family","T1")[:2],"add")
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
    if kind in ("add","decal"):
        host=list(amap.keys())
        new=[w for vals in amap.values() for w in vals] or list(act.keys())
    elif kind in ("recolor","material"):
        host=list(amap.keys())
        new=[w for vals in amap.values() for w in vals] or list(act.keys())
    else:
        new=list(act.keys()); host=list(amap.keys())
    return ",".join(new) if new else None, ",".join(host) if host else None
def argv_for(e, od, P):
    t=e["target_prompt"]
    a=["--image",str((PROJ/e["image"]).resolve()),"--source-prompt",e["source_prompt"],"--prompt",t,
      "--output",str(od/"result.png"),"--metadata-output",str(od/"metadata.json"),"--stats-output",str(od/"stats.json"),"--mask-output-dir",str(od/"masks"),
      "--method","dece_rf_flux","--seed",SEED,"--num-inference-steps","28","--n-max","24","--max-image-size","512","--max-sequence-length","512",
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
      "--true-cfg","--distilled-guidance","1.0"]
    if P["core"] is not None: a+=["--fixed-core-from-attention","--fixed-core-attention-percentile",str(P["core"])]
    new_tokens,host_tokens=pp_tokens(e, kind_of(e))
    if new_tokens: a+=["--new-tokens",new_tokens]
    if host_tokens: a+=["--host-tokens",host_tokens]
    if P["operation"]=="recolor":
        target=recolor_target(e)
        if target: a+=["--recolor-target",target]
        a+=["--minimal-recolor-scale","0.12","--final-recolor-blend-scale","0.45","--recolor-clean-projection-scale","0.20"]
    mask=local_mask(e)
    if mask:
        a+=["--semantic-base-mask",mask,"--support-control-mode","fixed","--support-external-mask-role","attention"]
    _adapt = __import__("os").environ.get("ADAPT_OVERRIDE_JSON")
    if _adapt:
        for _fl, _vl in json.loads(_adapt).get(kind_of(e), {}).items():
            if _fl in a: a[a.index(_fl) + 1] = str(_vl)
            else: a += [_fl, str(_vl)]
    return a
dev=torch.device("cuda"); 
e0=man[0]; od0=OUT/e0["key"]/"dece_rf_flux"/f"seed_{SEED}"; od0.mkdir(parents=True,exist_ok=True)
base=build_parser().parse_args(argv_for(e0,od0,KIND[kind_of(e0)]))
t0=time.time(); PIPE=load_flux_pipeline(base,dev); print("[ka-flux] loaded once %.1fs"%(time.time()-t0),flush=True)
flux_hrec.load_flux_pipeline=lambda a,d=dev: PIPE
done=0
for e in man[:LIMIT]:
    k=kind_of(e); P=KIND[k]; od=OUT/e["key"]/"dece_rf_flux"/f"seed_{SEED}"; od.mkdir(parents=True,exist_ok=True)
    args=build_parser().parse_args(argv_for(e,od,P)); ts=time.time()
    try:
        res=HRecFluxEdit(args); res.images[0].save(args.output)
        json.dump(res.metadata,open(args.metadata_output,"w")); json.dump(res.stats,open(args.stats_output,"w"))
        done+=1; print("OK",e["key"],k,"%.1fs"%(time.time()-ts),flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FAILED",e["key"],repr(ex),flush=True)
print("ALLDONE kindaware-flux-kindonly",done,"/",min(LIMIT,len(man)),flush=True)
