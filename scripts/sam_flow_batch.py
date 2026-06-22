import os, sys, json, shutil, time
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
SAM=PROJ/"_baselines/src/Sam-Flow"; sys.path.insert(0, str(SAM)); os.chdir(str(SAM))
from sam_flow.project_utils import case_output_path, case_scout_path, load_yaml, make_single_case, prepare_case_dirs
MODE=os.environ["MODE"]
man=json.load(open(PROJ/"data/pie_pilot_20260618/manifest.json"))
toks=json.load(open(PROJ/"data/pie_pilot_20260618/samflow_tokens.json"))
LIMIT=int(os.environ.get("LIMIT", len(man)))
OUT=PROJ/os.environ.get("OUT","outputs/pie15_baselines_20260618")
config=load_yaml(str(SAM/f"configs/{MODE}.yaml"))
config.setdefault("results",{})["root"]=str(OUT/"_samflow_tmp"/MODE)
config.setdefault("run",{})["overwrite"]=True
if MODE=="flux":
    from sam_flow.flowedit_flux import generate_scout_image_flux as gen, load_flux_pipeline as loadp
    from sam_flow.sam_flow_flux import run_sam_flow_flux_case as runcase
else:
    from sam_flow.flowedit_sd3 import generate_scout_image_sd3 as gen, load_sd3_pipeline as loadp
    from sam_flow.sam_flow_sd3 import run_sam_flow_sd3_case as runcase
tl=time.time(); pipe=loadp(config); print("[sf-batch] pipe loaded once %.1fs"%(time.time()-tl),flush=True)
done=0
for e in man[:LIMIT]:
    key=e["key"]; final=OUT/key/("sam_flow_"+MODE)/"seed_10"/"result.png"
    if final.exists(): print("skip",key); continue
    final.parent.mkdir(parents=True,exist_ok=True)
    t=toks[key]
    try:
        t0=time.time()
        case=make_single_case(image_path=str((PROJ/e["image"]).resolve()),source_prompt=e["source_prompt"],
            target_prompt=e["target_prompt"],source_mask_tokens=t["source"],target_mask_tokens=t["target"],
            unchanged_tokens=[],target_code=key)
        image_dir, case_dir = prepare_case_dirs(config["results"]["root"], case["image_name"], case["target_code"])
        scout_path = gen(pipe, case, config, image_dir, case_dir)
        out_path = runcase(pipe, case, config, scout_path, case_dir)
        shutil.copy(str(out_path), str(final)); done+=1; print("OK",key,"%.1fs"%(time.time()-t0),flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FAILED",key,repr(ex),flush=True)
print("ALLDONE sam_flow",MODE,done,"/",min(LIMIT,len(man)),flush=True)
