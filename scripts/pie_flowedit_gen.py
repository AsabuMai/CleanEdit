import json, yaml, os, os
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
man=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/pie_pilot_20260618/manifest.json"))))
repo=PROJ/"_baselines/src/FlowEdit"
# dataset yaml entries
ds=[]
for e in man:
    ds.append({"input_img": str((PROJ/e["image"]).resolve()),
               "source_prompt": e["source_prompt"],
               "target_prompts": [e["target_prompt"]],
               "target_codes": [e["key"]]})
(repo/"Data").mkdir(exist_ok=True)
yaml.safe_dump(ds, open(repo/"Data/pie15_dataset.yaml","w"), sort_keys=False, allow_unicode=True)
# exp yaml: SD3
exp=[{"exp_name":"FlowEdit_SD3_pie15","dataset_yaml":"Data/pie15_dataset.yaml","model_type":"SD3",
      "sampler_type":"FlowEditSD3","T_steps":28,"n_avg":1,"src_guidance_scale":3.5,
      "tar_guidance_scale":13.5,"n_min":0,"n_max":33,"seed":10}]
yaml.safe_dump(exp, open(repo/"pie15_exp.yaml","w"), sort_keys=False, allow_unicode=True)
print("wrote", repo/"Data/pie15_dataset.yaml", "and pie15_exp.yaml; n=",len(ds))
