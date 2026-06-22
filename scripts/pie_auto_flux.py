import json, subprocess, sys
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
PY=str(PROJ/".venv-h100/bin/python")
man=json.load(open(PROJ/"data/pie_pilot_20260618/manifest.json"))
OUT=PROJ/"outputs/pie_auto_flux_20260618"; SEED="10"
P=dict(edit_hedit=1.0,edit_anchor=0.10,edit_region=0.15,edit_target=0.18,edit_source=0.0,
       local_target=0.7,region_transport=0.10,outside_lock=0.15,rec=0.30,struct=0.45,traj=0.15,
       preserve_budget=0.12,preserve_gain=2.0)
done=0
for e in man:
    od=OUT/e["key"]/"dece_rf_flux"/f"seed_{SEED}"
    if (od/"result.png").exists(): print("skip",e["key"]); sys.stdout.flush(); continue
    od.mkdir(parents=True,exist_ok=True)
    cmd=[PY,str(PROJ/"scripts/run_dece_image.py"),
      "--backend","flux","--project",str(PROJ),"--python",PY,
      "--model-id","black-forest-labs/FLUX.1-dev",
      "--cache-dir",str(PROJ/".cache/huggingface/hub"),"--local-files-only","--model-offload",
      "--image",e["image"],"--source-prompt",e["source_prompt"],"--target-prompt",e["target_prompt"],
      "--method","dece_rf_flux","--seed",SEED,
      "--num-inference-steps","28","--n-max","24","--max-image-size","512","--max-sequence-length","512",
      "--src-guidance-scale","1.0","--base-guidance-scale","1.0","--tar-guidance-scale","5.0",
      "--support-control-mode","operation","--use-flux-attention-support",
      "--edit-operation","add_object","--support-relation","on_surface","--mask-layering-mode","object_contact",
      "--edit-hedit-guidance-scale",str(P["edit_hedit"]),"--edit-guidance-scale",str(P["edit_anchor"]),
      "--edit-region-guidance-scale",str(P["edit_region"]),"--edit-target-guidance-scale",str(P["edit_target"]),
      "--edit-source-guidance-scale",str(P["edit_source"]),
      "--edit-local-target-prompt",e["target_prompt"],"--edit-local-target-guidance-scale",str(P["local_target"]),
      "--edit-local-target-cfg-scale","5.0",
      "--rec-guidance-scale",str(P["rec"]),"--struct-guidance-scale",str(P["struct"]),
      "--trajectory-preserve-scale",str(P["traj"]),
      "--beta-max","1.0","--rec-stop-timestep","0.08","--linear-path-t-min","0.05",
      "--adaptive-clean-control","--adaptive-edit-target-rms","0.42","--adaptive-rmsgap-mode","legacy",
      "--adaptive-preserve-drift-budget",str(P["preserve_budget"]),"--adaptive-edit-gain","2.0",
      "--adaptive-preserve-gain",str(P["preserve_gain"]),
      "--adaptive-edit-weight-min","0.85","--adaptive-edit-weight-max","1.55",
      "--adaptive-preserve-weight-min","1.0","--adaptive-preserve-weight-max","1.65",
      "--adaptive-projection-scale","0.65","--adaptive-preserve-clean-correction-scale","0.5",
      "--region-target-transport-scale",str(P["region_transport"]),
      "--region-target-outside-lock-scale",str(P["outside_lock"]),
      "--final-postprocess-mode","mask_blend","--final-mask-blend-scale","1.0","--final-mask-alpha-gamma","1.0",
      "--mask-output-dir",str(od/"masks"),"--output",str(od/"result.png"),
      "--metadata-output",str(od/"metadata.json"),"--stats-output",str(od/"stats.json"),
      "--extra-arg=--true-cfg","--extra-arg=--distilled-guidance 1.0"]
    print("RUN-FLUXauto",e["key"]); sys.stdout.flush()
    r=subprocess.run(cmd)
    done+= (r.returncode==0)
    if r.returncode!=0: print("FAILED",e["key"],r.returncode); sys.stdout.flush()
print("ALLDONE",done,"/",len(man))
