import json
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
SRC = PROJ / "data/flowedit_compatible_135/manifest_sam_17new.json"
OUT = PROJ / "data/flowedit_compatible_135/manifest_flux_t2_tune7.json"
KEYS = {
    "fe_036_cake_red_blueberries_2_raspberries",
    "fe_198_piece_of_cake_1_cherry_on_top",
    "fe_201_pizza_1_pineapple_ham",
    "fe_205_pizza_slice_1_pepperoni",
    "fe_206_pizza_slice_2_mushrooms",
    "fe_207_pizza_tomato_olive_1_pepperoni",
    "fe_208_pizza_tomato_olive_2_mushrooms",
}


def main() -> None:
    items = [item for item in json.load(SRC.open(encoding="utf-8")) if item["key"] in KEYS]
    OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT, len(items), [item["key"] for item in items])


if __name__ == "__main__":
    main()
