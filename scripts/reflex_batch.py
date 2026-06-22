import os, sys, json, copy, glob, shutil, time
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
SRC=PROJ/"_baselines/src/ReFlex"
os.chdir(str(SRC)); sys.path.insert(0, str(SRC))
man=json.load(open(PROJ/"data/pie_pilot_20260618/manifest.json"))
LIMIT=int(os.environ.get("LIMIT", len(man)))
OUT=PROJ/os.environ.get("OUT","outputs/pie15_baselines_20260618")
e0=man[0]
sys.argv=["img_edit.py","--gpu","0","--seed","10","--img_path",str((PROJ/e0["image"]).resolve()),
  "--source_prompt",e0["source_prompt"],"--target_prompt",e0["target_prompt"],
  "--results_dir",str(OUT/"_tmp"),"--feature_steps","5","--attn_topk","5"]
import img_edit
_orig=img_edit.get_flux_pipeline; _c={}
def _gp(*a, **k):
    if "p" not in _c: _c["p"]=_orig(*a, **k); print("[reflex-batch] pipe loaded once",flush=True)
    return _c["p"]
img_edit.get_flux_pipeline=_gp
base=img_edit.args
done=0
for e in man[:LIMIT]:
    final=OUT/e["key"]/"reflex"/"seed_10"/"result.png"
    if final.exists(): print("skip",e["key"]); continue
    final.parent.mkdir(parents=True,exist_ok=True)
    rd=final.parent/"raw"
    a=copy.deepcopy(base)
    a.img_path=str((PROJ/e["image"]).resolve()); a.source_prompt=e["source_prompt"]; a.target_prompt=e["target_prompt"]
    a.results_dir=str(rd); a.seed=10
    t0=time.time()
    try:
        img_edit.main(a)
        found=glob.glob(str(rd/"**"/"target_0.png"),recursive=True)
        if found: shutil.copy(found[0],final); done+=1; print("OK",e["key"],"%.1fs"%(time.time()-t0),flush=True)
        else: print("FAILED-nooutput",e["key"],flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FAILED",e["key"],repr(ex),flush=True)
print("ALLDONE reflex",done,"/",min(LIMIT,len(man)),flush=True)
