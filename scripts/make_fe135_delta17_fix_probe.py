import json
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
SRC = PROJ / "data/flowedit_compatible_135/manifest_sam_17new.json"
OUT = PROJ / "data/flowedit_compatible_135/manifest_sam_delta17_fix2.json"
KEYS = {
    "fe_036_cake_red_blueberries_2_raspberries",
    "fe_211_puppies_3_puppets",
}


def main() -> None:
    items = [item for item in json.load(SRC.open(encoding="utf-8")) if item["key"] in KEYS]
    OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT, len(items), [item["key"] for item in items])


if __name__ == "__main__":
    main()
