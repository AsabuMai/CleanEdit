import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_sam_135.json"
OUT_DIR = PROJ / "outputs/fe135_full_dece_flux"


def thumb(path: Path, size: int, mask: bool = False) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if mask:
        image = ImageOps.colorize(ImageOps.grayscale(image), black="white", white=(255, 130, 0)).convert("RGB")
    image.thumbnail((size - 8, size - 8), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (size, size), "white")
    cell.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return cell


def result_path(key: str) -> Path:
    return OUT_DIR / key / "dece_rf_flux" / "seed_10" / "result.png"


def make_result_sheet(items: list[dict]) -> None:
    cols = 10
    size = 132
    label_h = 22
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, item in enumerate(items):
        key = item["key"]
        x = (idx % cols) * size
        y = (idx // cols) * (size + label_h)
        p = result_path(key)
        cell = thumb(p, size) if p.exists() else Image.new("RGB", (size, size), (245, 245, 245))
        sheet.paste(cell, (x, y + label_h))
        draw.text((x + 3, y + 4), key.replace("fe_", "")[:19], fill=(0, 0, 0))
    out = OUT_DIR / "_contact_results_135.jpg"
    sheet.save(out, quality=90)
    print(out)


def make_t2_sheet(items: list[dict]) -> None:
    t2 = [item for item in items if item.get("family_label") == "T2_container_insertion"]
    cols = ["Input", "Mask", "FLUX"]
    size = 170
    label_h = 48
    header_h = 24
    sheet = Image.new("RGB", (len(cols) * size + 20, len(t2) * (size + label_h) + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(cols):
        draw.text((10 + i * size + 6, 6), name, fill=(0, 0, 0))
    y = header_h
    for item in t2:
        key = item["key"]
        paths = [
            Path(item["image"]),
            PROJ / item["pp_local_mask"],
            result_path(key),
        ]
        for i, path in enumerate(paths):
            cell = thumb(path, size, mask=(i == 1)) if path.exists() else Image.new("RGB", (size, size), (245, 245, 245))
            x = 10 + i * size
            sheet.paste(cell, (x, y))
            draw.rectangle([x, y, x + size - 1, y + size - 1], outline=(220, 220, 220))
        draw.text((10, y + size + 5), key[:90], fill=(0, 0, 0))
        y += size + label_h
    out = OUT_DIR / "_contact_t2_insertions.jpg"
    sheet.save(out, quality=92)
    print(out)


def main() -> None:
    items = json.load(MANIFEST.open(encoding="utf-8"))
    make_result_sheet(items)
    make_t2_sheet(items)


if __name__ == "__main__":
    main()
