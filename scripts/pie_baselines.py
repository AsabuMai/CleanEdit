import json, subprocess, sys, os, shutil, glob
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project"); BASE=PROJ/"_baselines"; SRC=BASE/"src"
MAN=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/pie_dev_20260618/manifest.json"))))
LIMIT=int(os.environ.get("LIMIT", len(MAN)))
METHOD=os.environ["METHOD"]; OUTROOT=PROJ/os.environ.get("OUT","outputs/pie_baselines_20260618")
def _img(e): return str((PROJ/e["image"]).resolve())
def fireflow(e, od):
    return (SRC/"FireFlow/src", [str(BASE/"envs/fireflow-py310/bin/python"),"edit.py",
      "--source_prompt",e["source_prompt"],"--target_prompt",e["target_prompt"],"--guidance","2",
      "--source_img_dir",_img(e),"--num_steps","8","--inject","1","--start_layer_index","0",
      "--end_layer_index","37","--name","flux-dev","--sampling_strategy","fireflow",
      "--output_prefix",e["key"],"--output_dir",str(od),"--feature_path",str(od/"features"),"--offload","--seed","10"],
      lambda: sorted(glob.glob(str(od/"*.png"))+glob.glob(str(od/"*.jpg"))))
def rfsolver(e, od):
    return (SRC/"RF-Solver-Edit/FLUX_Image_Edit/src", [str(BASE/"envs/rf-solver-edit-py310/bin/python"),"edit.py",
      "--source_prompt",e["source_prompt"],"--target_prompt",e["target_prompt"],"--guidance","2",
      "--source_img_dir",_img(e),"--num_steps","25","--inject","3","--name","flux-dev",
      "--output_dir",str(od),"--feature_path",str(od/"features"),"--offload"],
      lambda: sorted(glob.glob(str(od/"*.png"))+glob.glob(str(od/"*.jpg"))))
def reflex(e, od):
    return (PROJ, [str(BASE/"envs/reflex-py310/bin/python"),str(SRC/"ReFlex/img_edit.py"),
      "--gpu","0","--seed","10","--img_path",_img(e),"--source_prompt",e["source_prompt"],
      "--target_prompt",e["target_prompt"],"--results_dir",str(od),"--feature_steps","5","--attn_topk","5"],
      lambda: glob.glob(str(od/"**/target_0.png"),recursive=True)+glob.glob(str(od/"target_0.png")))
BUILD={"fireflow":fireflow,"rf_solver_edit":rfsolver,"reflex":reflex}
build=BUILD[METHOD]; done=0
for e in MAN[:LIMIT]:
    final=OUTROOT/e["key"]/METHOD/"seed_10"/"result.png"
    if final.exists(): print("skip",e["key"]); continue
    od=OUTROOT/e["key"]/METHOD/"seed_10"/"raw"; od.mkdir(parents=True,exist_ok=True)
    cwd,cmd,locate=build(e,od)
    print("RUN",METHOD,e["key"]); sys.stdout.flush()
    r=subprocess.run(cmd,cwd=str(cwd))
    res=locate()
    if r.returncode==0 and res:
        shutil.copy(res[0], final); done+=1; print("OK",e["key"],"->",res[0])
    else: print("FAILED",e["key"],"rc",r.returncode,"found",res); sys.stdout.flush()
print("ALLDONE",METHOD,done,"/",min(LIMIT,len(MAN)))
