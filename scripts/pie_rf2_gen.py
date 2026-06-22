import json, yaml, os
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
man=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/pie_pilot_20260618/manifest.json"))))
ds=[{"input_img": str((PROJ/e["image"]).resolve()),
     "source_prompt": e["source_prompt"],
     "target_prompts": [e["target_prompt"]],
     "target_codes": [e["key"]]} for e in man]
# OT-RF
otr=PROJ/"_baselines/src/OT-RF"
yaml.safe_dump(ds, open(otr/"pie15_dataset.yaml","w"), sort_keys=False, allow_unicode=True)
yaml.safe_dump([{"exp_name":"OTRF_SD3_pie15","model_type":"SD3","T_steps":50,"n_avg":1,
  "src_guidance_scale":3.5,"tar_guidance_scale":13.5,"n_min":0,"n_max":33,"seed":10,
  "dataset_yaml":"pie15_dataset.yaml"}], open(otr/"pie15_exp.yaml","w"), sort_keys=False, allow_unicode=True)
# DRFS
drfs=PROJ/"_baselines/src/DeltaRectifiedFlowSampling"
yaml.safe_dump(ds, open(drfs/"pie15_dataset.yaml","w"), sort_keys=False, allow_unicode=True)
yaml.safe_dump([{"exp_name":"DRFS_SD3_pie15","dataset_yaml":"pie15_dataset.yaml","model_type":"SD3",
  "T_steps":50,"B":1,"src_guidance_scale":6,"tgt_guidance_scale":16.5,"num_steps":50,"seed":10,
  "eta":1.0,"scheduler_strategy":"descending","lr":"custom","optimizer":"SGD"}],
  open(drfs/"pie15_exp.yaml","w"), sort_keys=False, allow_unicode=True)
print("wrote OT-RF + DRFS yamls n=",len(ds))
