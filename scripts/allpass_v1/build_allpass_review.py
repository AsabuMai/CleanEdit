from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = ROOT / "data/flowedit_compatible_135/manifest_sam_135.json"
REGISTRY = ROOT / "data/flowedit_compatible_135/repair_allpass_v1/registry.json"
STAGE = ROOT / "tmp_allpass_v1_review"
THUMB = 92

METHODS = [
    ("source", "src"),
    ("old_sd3", "old SD3"),
    ("fixed_sd3", "fixed SD3"),
    ("old_flux", "old FLUX"),
    ("fixed_flux", "fixed FLUX"),
    ("sam_flow_sd3", "SAM-SD3"),
    ("sam_flow_flux", "SAM-FLUX"),
    ("reflex", "ReFLEx"),
    ("fireflow", "FireFlow"),
    ("rf_solver_edit", "RF-Solver"),
    ("otrf", "OT-RF"),
    ("drfs", "DRFS"),
]


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def first_glob(pattern: str) -> Path | None:
    hits = sorted(ROOT.glob(pattern))
    return hits[0] if hits else None


def right_half_if_drfs(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if "DeltaRectifiedFlowSampling" in str(path) and im.width >= im.height * 2:
        im = im.crop((im.width // 2, 0, im.width, im.height))
    return im


def result_path(method: str, entry: dict, registry: dict) -> Path | None:
    key = entry["key"]
    item = registry[key]
    if method == "source":
        return resolve(entry["image"])
    if method == "old_sd3":
        return first_glob(f"outputs/fe135_subjectpreserve_full_v10_sd3/{key}/*/seed_10/result.png")
    if method == "old_flux":
        return first_glob(f"outputs/fe135_full_dece_flux_v11c_pcie8_h100/{key}/*/seed_10/result.png")
    if method == "fixed_sd3":
        return ROOT / "outputs/fe135_allpass_sd3_v1" / key / item["sd3_recipe"] / "seed_10/result.png"
    if method == "fixed_flux":
        return ROOT / "outputs/fe135_allpass_flux_v1" / key / item["flux_recipe"] / "seed_10/result.png"
    if method in {"sam_flow_sd3", "sam_flow_flux", "reflex"}:
        return first_glob(f"outputs/flowedit135_baselines/{method}/{key}/*/seed_10/result.png")
    if method in {"fireflow", "rf_solver_edit"}:
        return first_glob(f"outputs/baselines/{method}/{key}/seed_10/result.png") or first_glob(
            f"outputs/baselines/{method}/{key}/*/seed_10/result.png"
        )
    if method == "otrf":
        return first_glob(f"_baselines/src/OT-RF/outputs/OTRF_SD3_FE135_ENH/SD3/src_{key}/tar_0/enhanced/*.png")
    if method == "drfs":
        return first_glob(f"_baselines/src/DeltaRectifiedFlowSampling/outputs/DRFS_SD3_FE135/SD3/src_{key}/tgt_0/*.png")
    return None


def save_thumb(path: Path, out: Path) -> None:
    im = right_half_if_drfs(path)
    im.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (THUMB, THUMB), "white")
    canvas.paste(im, ((THUMB - im.width) // 2, (THUMB - im.height) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)


def thumb_for(key: str, method: str) -> Path:
    return STAGE / "thumbs" / key / f"{method}.jpg"


def build_thumbs(manifest: list[dict], registry: dict) -> list[dict]:
    rows = []
    for entry in manifest:
        key = entry["key"]
        item = registry[key]
        for method, label in METHODS:
            path = result_path(method, entry, registry)
            status = "ok" if path and path.exists() else "missing"
            thumb = ""
            if status == "ok":
                t = thumb_for(key, method)
                save_thumb(path, t)
                thumb = str(t.relative_to(STAGE))
            rows.append(
                {
                    "key": key,
                    "family_label": entry.get("family_label", ""),
                    "category": item["category"],
                    "priority": item["priority"],
                    "method": method,
                    "label": label,
                    "status": status,
                    "path": str(path) if path else "",
                    "thumb": thumb,
                }
            )
    return rows


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, max_chars: int = 44) -> None:
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def make_family_sheet(family: str, tasks: list[dict], registry: dict) -> None:
    out_dir = STAGE / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    left = 250
    pad = 8
    header = 44
    row_h = THUMB + 38
    width = left + len(METHODS) * (THUMB + pad) + pad
    height = header + len(tasks) * row_h + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 8), f"{family} n={len(tasks)} allpass_v1", fill=(0, 0, 0), font=font)
    for j, (_, label) in enumerate(METHODS):
        draw_text(draw, (left + j * (THUMB + pad), 24), label, font, 14)
    for i, entry in enumerate(tasks):
        key = entry["key"]
        item = registry[key]
        y = header + i * row_h
        draw_text(draw, (pad, y + 4), key, font, 42)
        draw_text(draw, (pad, y + 20), item["category"], font, 42)
        for j, (method, _) in enumerate(METHODS):
            x = left + j * (THUMB + pad)
            box = (x, y + 36, x + THUMB, y + 36 + THUMB)
            t = thumb_for(key, method)
            draw.rectangle(box, fill=(242, 242, 242), outline=(210, 210, 210))
            if t.exists():
                im = Image.open(t).convert("RGB")
                canvas.paste(im, (x, y + 36))
            else:
                draw.text((x + 14, y + 36 + THUMB // 2), "MISS", fill=(120, 120, 120), font=font)
            draw.rectangle(box, outline=(190, 190, 190))
    canvas.save(out_dir / f"{family}.jpg", quality=92)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    rows = build_thumbs(manifest, registry)
    with open(STAGE / "allpass_coverage.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    families: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest:
        families[entry.get("family_label", "unknown")].append(entry)
    for family, tasks in sorted(families.items()):
        make_family_sheet(family, tasks, registry)
    print(STAGE)


if __name__ == "__main__":
    main()
