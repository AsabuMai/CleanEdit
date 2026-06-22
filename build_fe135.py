import json, os, yaml
from pathlib import Path

PROJ = Path("/cluster/users/grad/2025/25t8103/project")
FEIMG = PROJ / "_baselines/src/FlowEdit/Data/Images"
FEYAML = PROJ / "_baselines/src/FlowEdit/Data/flowedit.yaml"
D123 = PROJ / "data/flowedit_compatible_123"
D135 = PROJ / "data/flowedit_compatible_135"
UNIQ = D135 / "unique_inputs"
UNIQ.mkdir(parents=True, exist_ok=True)

# ---- 12 expanded cases: (key, init_img_name, target_code, family_label, config, host, new) ----
EXPANDED = [
    ("fe_090_dog_2_puppet",                  90,  "dog",                "2_puppet",         "T5_same_color_material", "7_material_transfer_60", "dog",        "puppet"),
    ("fe_091_dog_3_crochet",                 91,  "dog",                "3_crochet",        "T5_same_color_material", "7_material_transfer_60", "dog",        "crochet"),
    ("fe_138_horse_2_pink_toy_horse",        138, "horse",              "2_pink_toy_horse", "T5_same_color_material", "7_material_transfer_60", "horse",      "pink,toy"),
    ("fe_211_puppies_3_puppets",             211, "puppies",            "3_puppets",        "T5_same_color_material", "7_material_transfer_60", "puppies",    "puppets"),
    ("fe_252_tiger_2_crochet_tiger",         252, "tiger",              "2_crochet_tiger",  "T5_same_color_material", "7_material_transfer_60", "tiger",      "crochet"),
    ("fe_036_cake_red_blueberries_2_raspberries", 36, "cake_red_blueberries", "2_raspberries", "T2_container_insertion", "2_container_insertion_80", "blueberries", "raspberries"),
    ("fe_198_piece_of_cake_1_cherry_on_top", 198, "piece_of_cake",      "1_cherry_on_top",  "T2_container_insertion", "2_container_insertion_80", "cake",       "cherry"),
    ("fe_201_pizza_1_pineapple_ham",         201, "pizza",              "1_pineapple_ham",  "T2_container_insertion", "2_container_insertion_80", "pizza",      "pineapple,ham"),
    ("fe_205_pizza_slice_1_pepperoni",       205, "pizza_slice",        "1_pepperoni",      "T2_container_insertion", "2_container_insertion_80", "pizza",      "pepperoni"),
    ("fe_206_pizza_slice_2_mushrooms",       206, "pizza_slice",        "2_mushrooms",      "T2_container_insertion", "2_container_insertion_80", "pizza",      "mushrooms"),
    ("fe_207_pizza_tomato_olive_1_pepperoni",207, "pizza_tomato_olive", "1_pepperoni",      "T2_container_insertion", "2_container_insertion_80", "pizza",      "pepperoni"),
    ("fe_208_pizza_tomato_olive_2_mushrooms",208, "pizza_tomato_olive", "2_mushrooms",      "T2_container_insertion", "2_container_insertion_80", "pizza",      "mushrooms"),
]
FAMLBL2F = {"T5_same_color_material": "F5", "T2_container_insertion": "F2"}

# ---- load raw flowedit.yaml, index by init_img name ----
raw = yaml.safe_load(open(FEYAML))
byname = {}
for e in raw:
    nm = e.get("init_img", "").split("/")[-1].rsplit(".", 1)[0]
    byname[nm] = e

def link(target, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(target, dst)

# ---- 1. fix meditation symlinks (meditation1 -> meditation2) ----
med2 = FEIMG / "meditation2.png"
fixed_med = []
for k in ["fe_172_meditation1_1_sand_sculpture", "fe_173_meditation1_2_golden_buddha_statue",
          "fe_174_meditation1_3_wooden_buddha_statue", "fe_175_meditation1_4_golden_statue",
          "fe_176_meditation1_5_wooden_statue"]:
    link(med2, UNIQ / (k + ".png"))
    fixed_med.append(k)

# ---- 2. build expanded entries + their unique_inputs links ----
exp_entries = []
for key, idx, nm, code, famlbl, cfg, host, new in EXPANDED:
    fe = byname[nm]
    codes = fe["target_codes"]
    i = codes.index(code)
    tgt = fe["target_prompts"][i].strip()
    img = FEIMG / (nm + ".png")
    link(img, UNIQ / (key + ".png"))
    exp_entries.append({
        "key": key, "flowedit_idx": idx, "flowedit_status": "expanded-stress",
        "flowedit_reason": "expanded_clean_sample_close_to_T2_T5_def",
        "config": cfg, "family": FAMLBL2F[famlbl], "family_label": famlbl,
        "image": str(img), "source_prompt": fe["source_prompt"].strip(), "target_prompt": tgt,
        "global_mask": True, "host_tokens": host, "new_tokens": new,
        "pp_aspect_mapping": {host: new.split(",")},
        "subset": "expanded",
    })

# ---- 3. fixed 123 entries (meditation image path -> meditation2), tag subset=strict ----
m123 = json.load(open(D123 / "manifest.json"))
for e in m123:
    if e["key"] in fixed_med:
        e["image"] = str(med2)
    e["subset"] = "strict"

# ---- copy/link all 123 unique_inputs into 135 (reuse existing links, but meditation now fixed) ----
src_uniq = D123 / "unique_inputs"
for e in m123:
    dst = UNIQ / (e["key"] + ".png")
    if e["key"] in fixed_med:
        continue  # already linked to meditation2 above
    s = src_uniq / (e["key"] + ".png")
    real = os.path.realpath(s)
    link(Path(real), dst)

# ---- 4. write manifest_135 (strict 123 + expanded 12) and delta-17 ----
m135 = m123 + exp_entries
json.dump(m135, open(D135 / "manifest.json", "w"), indent=2, ensure_ascii=False)

delta17 = [e for e in m135 if e["key"] in set(fixed_med) | set(x[0] for x in EXPANDED)]
json.dump(delta17, open(D135 / "manifest_17new.json", "w"), indent=2, ensure_ascii=False)

# point delta image to the unique_inputs link path (stable) for runners that use 'image'
for e in delta17:
    e["image"] = str(UNIQ / (e["key"] + ".png"))
json.dump(delta17, open(D135 / "manifest_17new_uniq.json", "w"), indent=2, ensure_ascii=False)

# ---- 5. baseline dataset yaml for the 17 new (flowedit input format) ----
ds = []
for e in delta17:
    ds.append({"input_img": str(UNIQ / (e["key"] + ".png")),
               "source_prompt": e["source_prompt"],
               "target_prompts": [e["target_prompt"]],
               "target_codes": [e["key"]]})
yaml.safe_dump(ds, open(D135 / "flowedit17_dataset.yaml", "w"), sort_keys=False, allow_unicode=True, default_flow_style=False)

# ---- verify all images exist ----
miss = [e["key"] for e in m135 if not os.path.exists(e["image"])]
print("manifest_135 N=", len(m135), "| expanded=", len(exp_entries), "| delta17=", len(delta17))
print("missing images:", miss)
print("families in 135:", {f: sum(1 for e in m135 if e["family_label"] == f) for f in sorted(set(e["family_label"] for e in m135))})
print("subsets:", {s: sum(1 for e in m135 if e.get("subset") == s) for s in ["strict", "expanded"]})
