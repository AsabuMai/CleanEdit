import json
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_118/manifest.json"
OUT = PROJ / "data/flowedit_compatible_118/manifest_samflow_sd3_retry.json"

HOST_FIX = {
    "fe_017_boat_silhouette_1_sailboat_white_sails_red_hull": "boat",
    "fe_024_bus_2_volkswagen_logo": "bus",
    "fe_027_butterflies_1_yellow": "butterflies",
    "fe_084_cupcake_2_red_velvet": "cupcake",
    "fe_114_flowers_1_orange_yellow_white": "flowers",
    "fe_115_flowers_2_blue_purple_white": "flowers",
    "fe_221_rocks_6_colorful_wooden_blocks": "rocks",
}


def main() -> None:
    items = json.load(MANIFEST.open())
    retry = []
    for item in items:
        if item["key"] not in HOST_FIX:
            continue
        item = dict(item)
        item["host_tokens"] = HOST_FIX[item["key"]]
        new_tokens = [x.strip() for x in item.get("new_tokens", "").split(",") if x.strip()]
        if new_tokens:
            item["pp_aspect_mapping"] = {item["host_tokens"]: new_tokens}
        retry.append(item)
    OUT.write_text(json.dumps(retry, indent=2) + "\n", encoding="utf-8")
    print("retry", len(retry), OUT)
    for item in retry:
        print(item["key"], item["host_tokens"], item.get("new_tokens", ""))


if __name__ == "__main__":
    main()
