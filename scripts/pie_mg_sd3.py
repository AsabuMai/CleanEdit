import os, sys, json, types, time
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
man=json.load(open(os.environ.get("MANIFEST",str(PROJ/"data/pie_pilot_20260618/manifest.json"))))
LIMIT=int(os.environ.get("LIMIT",len(man))); OUT=PROJ/os.environ.get("OUT","outputs/pie_kindaware_sd3_20260618"); SEED="10"
# SD3 per-kind params (from pretty_matrix command.txt)
KIND={"add":dict(hedit="0.65",text="0.08",rec="0.22",struct="0.45",core="1.35",subj="0.35",color=None),
      "decal":dict(hedit="0.65",text="0.08",rec="0.45",struct="0.45",core="1.35",subj="0.35",color=None),
      "material":dict(hedit="0.68",text="0.14",rec="0.40",struct="0.45",core="1.35",subj="0.35",color=None),
      "recolor":dict(hedit="0.18",text="0.02",rec="0.58",struct="0.45",core="1.35",subj="1.0",color="0.10")}
FAM2KIND={"T1":"add","T2":"add","T3":"decal","T4":"recolor","T5":"material"}
CFG2KIND={"1":"add","2":"add","3":"add","6":"recolor","7":"material"}
def kind_of(e):
    if "family" in e: return FAM2KIND.get(e["family"][:2],"add")
    return CFG2KIND.get(str(e.get("config","1"))[0],"add")
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
      "--object-mask-provider","semantic","--semantic-base-mask",str((PROJ/e["mask"]).resolve()),"--grounding-method","external_mask","--final-edit-mask",str((PROJ/e["mask"]).resolve()),"--final-edit-mask-mode","replace","--final-outside-restore-mask",str((PROJ/e["mask"]).resolve()),"--final-outside-restore-scale","1.0","--edit-operation","add_object","--relation","on_surface","--mask-layering-mode","object_contact",
      "--adaptive-clean-control","--adaptive-edit-target-rms","0.42","--adaptive-rmsgap-mode","legacy","--adaptive-preserve-drift-budget","0.12",
      "--adaptive-edit-gain","2.0","--adaptive-preserve-gain","4.2","--adaptive-edit-weight-min","0.85","--adaptive-edit-weight-max","1.55",
      "--adaptive-preserve-weight-min","1.0","--adaptive-preserve-weight-max","1.65","--adaptive-projection-scale","0.65","--adaptive-preserve-clean-correction-scale","0.5",
      "--photo-prompt-mode","both","--log-every","7"]
    if P["color"] is not None: argv+=["--edit-color-guidance-scale",P["color"]]
    sys.argv=["run_edit_sd3.py"]+argv; ts=time.time()
    try: run_edit_sd3.main(); done+=1; print("OK",e["key"],k,"%.1fs"%(time.time()-ts),flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FAILED",e["key"],repr(ex),flush=True)
print("ALLDONE kindaware-sd3",done,"/",min(LIMIT,len(man)),flush=True)
