import json, os, sys, subprocess, glob, shutil
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
SAM=PROJ/"_baselines/src/Sam-Flow"; PY=str(PROJ/"_baselines/envs/sam-flow-py310/bin/python")
MODE=os.environ["MODE"]
man=json.load(open(PROJ/"data/pie_pilot_20260618/manifest.json"))
toks=json.load(open(PROJ/"data/pie_pilot_20260618/samflow_tokens.json"))
OUT=PROJ/os.environ.get("OUT","outputs/pie15_baselines_20260618")
done=0
for e in man:
    key=e["key"]; final=OUT/key/("sam_flow_"+MODE)/"seed_10"/"result.png"
    if final.exists(): print("skip",key); sys.stdout.flush(); continue
    oroot=final.parent/"raw"; oroot.mkdir(parents=True,exist_ok=True)
    img=str((PROJ/e["image"]).resolve()); tc=key
    cmd=[PY,"scripts/run_image.py","--mode",MODE,"--config",f"configs/{MODE}.yaml",
         "--image",img,"--source-prompt",e["source_prompt"],"--target-prompt",e["target_prompt"],
         "--target-code",tc,"--output-root",str(oroot),"--overwrite"]
    for t in toks[key]["source"]: cmd+=["--source-token",t]
    for t in toks[key]["target"]: cmd+=["--target-token",t]
    print("RUN samflow",MODE,key,flush=True)
    r=subprocess.run(cmd,cwd=str(SAM))
    found=glob.glob(str(oroot/"**"/f"{tc}_output.png"),recursive=True) or glob.glob(str(oroot/"**"/"*output*.png"),recursive=True) or glob.glob(str(oroot/"**"/"*.png"),recursive=True)
    if r.returncode==0 and found:
        shutil.copy(found[0],final); done+=1; print("OK",key,"->",found[0],flush=True)
    else: print("FAILED",key,r.returncode,found[:2],flush=True)
print("ALLDONE sam_flow",MODE,done,"/",len(man),flush=True)
