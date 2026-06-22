import json
from pathlib import Path

PROJ = Path("/cluster/users/grad/2025/25t8103/project")
SRC = PROJ / "data/flowedit_compatible_118/manifest_118_run.json"
DST = PROJ / "data/flowedit_compatible_118/manifest_repair_probe.json"

KEYS = [
    # T1 attached accessory: current runs preserve well but under-edit.
    "fe_046_cat_crown_1_black_top_hat",
    "fe_094_dog_6_red_top_hat",
    "fe_192_parrots_1_top_hat",
    # T2 insertion/topping.
    "fe_031_cake_1_berries",
    "fe_180_milk_4_whipped_cream",
    "fe_186_muffins_1_strawberries",
    # T3 surface decal/text: repair maps these to decal instead of add.
    "fe_116_free_wifi_1_free_beer",
    "fe_118_gas_station_1_cvpr",
    "fe_170_luna_7_heart",
    # T4 recolor sanity checks.
    "fe_000_bear_1_black_bear",
    "fe_027_butterflies_1_yellow",
    "fe_128_gray_bird_2_red_bird",
    # T5 material/medium transfer.
    "fe_021_brown_owl_3_origami_owl",
    "fe_127_gray_bird_1_origami_bird",
    "fe_171_meditation_1_wooden_statue",
]

items = json.load(SRC.open())
by_key = {item["key"]: item for item in items}
missing = [key for key in KEYS if key not in by_key]
if missing:
    raise SystemExit(f"missing keys: {missing}")
DST.write_text(json.dumps([by_key[key] for key in KEYS], indent=2) + "\n")
print("wrote", DST, len(KEYS))
