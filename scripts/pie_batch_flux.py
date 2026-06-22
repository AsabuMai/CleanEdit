import sys, json, os, time
from pathlib import Path
import torch
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
sys.path.insert(0, str(PROJ/"flux"))
import flux_hrec
from flux_hrec import build_parser, load_flux_pipeline, HRecFluxEdit

man=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/pie_pilot_20260618/manifest.json"))))
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else len(man)
OUT=PROJ/os.environ.get("FLUX_OUT","outputs/pie_batch_flux_20260618"); SEED="10"
def argv_for(e, od):
    t=e["target_prompt"]
    return ["--image",e["image"],"--source-prompt",e["source_prompt"],"--prompt",t,
      "--output",str(od/"result.png"),"--metadata-output",str(od/"metadata.json"),"--stats-output",str(od/"stats.json"),"--mask-output-dir",str(od/"masks"),
      "--method","dece_rf_flux","--seed",SEED,
      "--num-inference-steps","28","--n-max","24","--max-image-size","512","--max-sequence-length","512",
      "--src-guidance-scale","1.0","--base-guidance-scale","1.0","--tar-guidance-scale","5.0",
      "--support-control-mode","operation","--use-flux-attention-support",
      "--edit-operation","add_object","--support-relation","on_surface","--mask-layering-mode","object_contact",
      "--edit-hedit-guidance-scale",os.environ.get("EDIT_HEDIT","1.0"),"--edit-guidance-scale","0.10","--edit-region-guidance-scale","0.15",
      "--edit-target-guidance-scale",os.environ.get("EDIT_TARGET","0.18"),"--edit-source-guidance-scale","0.0",
      "--edit-local-target-prompt",t,"--edit-local-target-guidance-scale",os.environ.get("LOCAL_TARGET","0.7"),"--edit-local-target-cfg-scale","5.0",
      "--rec-guidance-scale","0.30","--struct-guidance-scale","0.45","--trajectory-preserve-scale","0.15",
      "--beta-max","1.0","--rec-stop-timestep","0.08","--linear-path-t-min","0.05",
      "--adaptive-clean-control","--adaptive-edit-target-rms","0.42","--adaptive-rmsgap-mode","legacy",
      "--adaptive-preserve-drift-budget","0.12","--adaptive-edit-gain","2.0","--adaptive-preserve-gain","2.0",
      "--adaptive-edit-weight-min","0.85","--adaptive-edit-weight-max","1.55","--adaptive-preserve-weight-min","1.0","--adaptive-preserve-weight-max","1.65",
      "--adaptive-projection-scale","0.65","--adaptive-preserve-clean-correction-scale","0.5",
      "--region-target-transport-scale","0.10","--region-target-outside-lock-scale","0.15",
      "--final-postprocess-mode","mask_blend","--final-mask-blend-scale","1.0","--final-mask-alpha-gamma","1.0",
      "--model-id","black-forest-labs/FLUX.1-dev","--cache-dir",str(PROJ/".cache/huggingface/hub"),"--local-files-only",
      "--true-cfg","--distilled-guidance","1.0"]
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
e0=man[0]; od0=OUT/e0["key"]/"dece_rf_flux"/f"seed_{SEED}"; od0.mkdir(parents=True,exist_ok=True)
base=build_parser().parse_args(argv_for(e0,od0))
t0=time.time(); PIPE=load_flux_pipeline(base, device); print("[batch] pipeline loaded once in %.1fs (offload=%s)"%(time.time()-t0, base.model_offload), flush=True)
flux_hrec.load_flux_pipeline = lambda a, d=device: PIPE   # reuse cached pipe
done=0
for e in man[:LIMIT]:
    od=OUT/e["key"]/"dece_rf_flux"/f"seed_{SEED}"; od.mkdir(parents=True,exist_ok=True)
    args=build_parser().parse_args(argv_for(e,od))
    ts=time.time(); res=HRecFluxEdit(args)
    res.images[0].save(args.output)
    json.dump(res.metadata, open(args.metadata_output,"w"), indent=2)
    json.dump(res.stats, open(args.stats_output,"w"), indent=2)
    done+=1; print("[batch] %s done in %.1fs"%(e["key"], time.time()-ts), flush=True)
print("ALLDONE",done,"/",LIMIT, flush=True)
