import json, subprocess, sys
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
PY=str(PROJ/".venv/bin/python")
man=json.load(open(PROJ/"data/pie_pilot_20260618/manifest.json"))
OUT=PROJ/"outputs/pie_pilot_sd3_20260618"; SEED="10"
done=0
for e in man:
    od=OUT/e["key"]/"support_v3_controller_rmsgap"/f"seed_{SEED}"
    if (od/"result.png").exists(): print("skip",e["key"]); sys.stdout.flush(); continue
    od.mkdir(parents=True,exist_ok=True)
    m=e["mask"]
    cmd=[PY,str(PROJ/"run_edit_sd3.py"),
      "--image",e["image"],"--source-prompt",e["source_prompt"],"--prompt",e["target_prompt"],
      "--output",str(od/"result.png"),"--stats-output",str(od/"stats.json"),
      "--metadata-output",str(od/"metadata.json"),"--mask-output-dir",str(od/"masks"),
      "--max-image-size","512","--seed",SEED,"--num-inference-steps","28","--n-max","24",
      "--src-guidance-scale","1.0","--base-guidance-scale","1.0","--tar-guidance-scale","10.5",
      "--edit-hedit-guidance-scale","0.68","--edit-guidance-scale","0.0","--edit-region-guidance-scale","0.0",
      "--edit-target-guidance-scale","0.0","--edit-source-guidance-scale","0.0",
      "--edit-text-guidance-scale","0.14","--edit-text-source-scale","0.8","--edit-text-core-weight","1.0","--edit-text-subject-weight","0.3",
      "--rec-guidance-scale","0.40","--struct-guidance-scale","0.45","--trajectory-preserve-scale","0.25","--trajectory-subject-preserve-scale","0.0",
      "--edit-core-scale","1.35","--edit-subject-scale","0.35",
      "--region-target-transport-scale","0.0","--region-target-outside-lock-scale","0.0",
      "--rec-stop-timestep","0.08","--beta-max","1.0","--velocity-conversion-mode","linear_path","--linear-path-t-min","0.05",
      "--object-mask-provider","semantic","--support-mask",m,"--grounding-method","external_mask",
      "--final-edit-mask",m,"--final-edit-mask-mode","replace",
      "--edit-operation","add_object","--relation","on_surface","--mask-layering-mode","object_contact",
      "--adaptive-clean-control","--adaptive-edit-target-rms","0.42","--adaptive-rmsgap-mode","legacy",
      "--adaptive-preserve-drift-budget","0.12","--adaptive-edit-gain","2.0","--adaptive-preserve-gain","4.2",
      "--adaptive-edit-weight-min","0.85","--adaptive-edit-weight-max","1.55","--adaptive-preserve-weight-min","1.0","--adaptive-preserve-weight-max","1.65",
      "--adaptive-projection-scale","0.65","--adaptive-preserve-clean-correction-scale","0.5",
      "--photo-prompt-mode","both","--log-every","7"]
    print("RUN-SD3",e["key"],e["family"]); sys.stdout.flush()
    r=subprocess.run(cmd)
    if r.returncode==0: done+=1
    else: print("FAILED",e["key"],r.returncode); sys.stdout.flush()
print("ALLDONE",done,"/",len(man))
