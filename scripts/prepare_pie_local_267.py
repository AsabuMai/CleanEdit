import json
import re
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

PROJ = Path("/cluster/users/grad/2025/25t8103/project")
OUT = PROJ / "data/pie_local_267_20260619"
RAW = OUT / "raw_rows"
IMG = OUT / "images"
MSK = OUT / "masks"

CONFIGS = [
    "2_add_object_80",
    "6_change_attribute_color_40",
    "7_change_attribute_material_40",
]
COUNTS = {
    "2_add_object_80": 80,
    "6_change_attribute_color_40": 40,
    "7_change_attribute_material_40": 40,
}
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=UB-CVML-Group/PIE_Bench_pp&config={}&split=V1&offset={}&length={}"
)


def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def write_mask(mask_text, path):
    values = [int(x) for x in re.findall(r"\d+", str(mask_text or ""))]
    image = Image.new("L", (512, 512), 0)
    pix = image.load()
    is_global = False
    if len(values) == 2 and values[0] == 0 and values[1] == 262144:
        image.paste(255, (0, 0, 512, 512))
        is_global = True
    elif len(values) >= 2:
        for start, end in zip(values[0::2], values[1::2]):
            start = max(0, min(262143, start))
            end = max(0, min(262144, end))
            if end < start:
                start, end = end, start
            for idx in range(start, end):
                pix[idx % 512, idx // 512] = 255
    image.save(path)
    return is_global


def load_rows(config):
    raw_path = RAW / f"{config}.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8"))
    rows = []
    total = COUNTS[config]
    for offset in range(0, total, 100):
        page = get_json(ROWS_URL.format(config, offset, min(100, total - offset)))
        rows.extend(page["rows"])
    raw_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    for directory in (RAW, IMG, MSK):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = []
    for config in CONFIGS:
        rows = load_rows(config)
        for item in rows:
            row = item["row"]
            row_id = row["id"]
            key = f"{config}__{row_id}"
            image_path = IMG / f"{key}.png"
            mask_path = MSK / f"{key}.png"

            image_info = row.get("image") or {}
            image_url = image_info.get("src") or image_info.get("path")
            if image_url and not image_path.exists():
                Image.open(BytesIO(get_bytes(image_url))).convert("RGB").save(image_path)

            if mask_path.exists():
                global_mask = str(row.get("mask", "")).strip() == "0 262144"
            else:
                global_mask = write_mask(row.get("mask"), mask_path)

            manifest.append(
                {
                    "key": key,
                    "id": row_id,
                    "config": config,
                    "image": str(image_path.relative_to(PROJ)),
                    "mask": str(mask_path.relative_to(PROJ)),
                    "global_mask": global_mask,
                    "source_prompt": row.get("source_prompt", ""),
                    "target_prompt": row.get("target_prompt", ""),
                    "edit_action": row.get("edit_action", ""),
                    "aspect_mapping": row.get("aspect_mapping", ""),
                    "blended_words": row.get("blended_words", ""),
                    "source_dataset": "UB-CVML-Group/PIE_Bench_pp",
                }
            )
        print(config, len(rows), flush=True)

    manifest.sort(key=lambda e: (e["config"], e["id"]))
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("manifest", len(manifest), flush=True)
    print("images", len(list(IMG.glob("*.png"))), flush=True)
    print("masks", len(list(MSK.glob("*.png"))), flush=True)


if __name__ == "__main__":
    main()
