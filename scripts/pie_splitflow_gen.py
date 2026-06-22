import json, yaml, os
from pathlib import Path
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
man=json.load(open(os.environ.get("MANIFEST", str(PROJ/"data/pie_pilot_20260618/manifest.json"))))
repo=PROJ/"_baselines/src/SplitFlow"
ds=[{"input_img": str((PROJ/e["image"]).resolve()),
     "source_prompt": [e["source_prompt"]],
     "target_prompts": [e["target_prompt"]],
     "target_codes": [e["key"]]} for e in man]
yaml.safe_dump(ds, open(repo/"pie15_dataset.yaml","w"), sort_keys=False, allow_unicode=True)
exp=[{"exp_name":"SplitFlow_SD3_pie15","dataset_yaml":"pie15_dataset.yaml","model_type":"SD3",
      "sampler_type":"FlowEditSD3","T_steps":50,"n_avg":1,"src_guidance_scale":3.5,
      "tar_guidance_scale":13.5,"n_min":0,"n_max":33,"seed":10}]
yaml.safe_dump(exp, open(repo/"pie15_exp.yaml","w"), sort_keys=False, allow_unicode=True)
print("wrote SplitFlow yamls n=",len(ds))
