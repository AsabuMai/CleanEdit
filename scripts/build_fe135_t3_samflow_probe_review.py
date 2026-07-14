from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REMOTE_PREFIX = "/cluster/users/grad/2025/25t8103/project/"
METHOD_LABELS = {
    "sam_flow_sd3": "Sam-Flow SD3 probe",
    "sam_flow_flux": "Sam-Flow FLUX probe",
}


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rel_project(path: str) -> Path:
    if path.startswith(REMOTE_PREFIX):
        return Path(path[len(REMOTE_PREFIX) :])
    return Path(path)


def map_project_path(project_tree: Path, path: str) -> Path:
    if not path:
        return Path()
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    return project_tree / rel_project(path)


def audit_by_pair(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["method"], row["key"]): row for row in rows}


def mask_bbox(mask_path: Path) -> tuple[int, int, int, int] | None:
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L")
    return mask.point(lambda p: 255 if p > 16 else 0).getbbox()


def expanded_bbox(
    bbox: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    w, h = image_size
    if bbox is None:
        side = min(w, h) * 0.62
        cx, cy = w / 2, h / 2
        x0, y0, x1, y1 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
    else:
        x0, y0, x1, y1 = bbox
        bw = max(8, x1 - x0)
        bh = max(8, y1 - y0)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        side = max(bw, bh) * scale
        x0, y0, x1, y1 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
    return (
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        min(w, int(round(x1))),
        min(h, int(round(y1))),
    )


def missing_tile(size: tuple[int, int], label: str) -> Image.Image:
    out = Image.new("RGB", size, (248, 225, 225))
    draw = ImageDraw.Draw(out)
    font = load_font(15)
    draw.text((8, 8), "MISSING", fill=(150, 0, 0), font=font)
    draw.text((8, 30), label[:32], fill=(100, 0, 0), font=font)
    return out


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        return missing_tile(size, path.name)
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, (246, 246, 246))
    out.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return out


def crop_fit(path: Path, crop_box: tuple[int, int, int, int], source_size: tuple[int, int], size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        return missing_tile(size, path.name)
    im = Image.open(path).convert("RGB")
    if im.size != source_size:
        sx = im.width / source_size[0]
        sy = im.height / source_size[1]
        box = (
            int(round(crop_box[0] * sx)),
            int(round(crop_box[1] * sy)),
            int(round(crop_box[2] * sx)),
            int(round(crop_box[3] * sy)),
        )
    else:
        box = crop_box
    crop = im.crop(box)
    crop.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, (246, 246, 246))
    out.paste(crop, ((size[0] - crop.width) // 2, (size[1] - crop.height) // 2))
    return out


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        nxt = f"{cur} {word}".strip()
        if not cur or draw.textlength(nxt, font=font) <= width:
            cur = nxt
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def make_sheet(
    *,
    out_path: Path,
    rows: list[dict[str, str]],
    manifest_by_key: dict[str, dict],
    audit: dict[tuple[str, str], dict[str, str]],
    project_tree: Path,
    output_root: Path,
    mode: str,
    part_index: int,
    part_count: int,
) -> None:
    columns = ("source", "current fail", "probe")
    cell = (260, 205) if mode == "overview" else (285, 285)
    label_h = 88
    top_h = 58
    row_h = cell[1] + label_h
    sheet = Image.new("RGB", (cell[0] * len(columns), top_h + row_h * len(rows)), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(20)
    head_font = load_font(15)
    label_font = load_font(14)
    draw.text((8, 8), f"FE135 T3 Sam-Flow probe {mode} part {part_index + 1}/{part_count}", fill=(0, 0, 0), font=title_font)
    for col, label in enumerate(columns):
        draw.text((col * cell[0] + 6, 34), label, fill=(0, 0, 0), font=head_font)
    y = top_h
    for row in rows:
        baseline = row["baseline"]
        key = row["task"]
        original_method = "sd3" if baseline == "sam_flow_sd3" else "flux"
        item = manifest_by_key[key]
        source = map_project_path(project_tree, item["image"])
        current = Path(audit[(original_method, key)]["result_path"])
        probe = map_project_path(project_tree, row.get("result_image", ""))
        if not probe:
            probe = project_tree / output_root / baseline / key / "seed_10" / "result.png"

        source_size = Image.open(source).size if source.exists() else (512, 512)
        bbox = mask_bbox(project_tree / item.get("pp_local_mask", ""))
        crop_box = expanded_bbox(bbox, source_size, scale=2.8)
        paths = (source, current, probe)
        for col, path in enumerate(paths):
            tile = fit(path, cell) if mode == "overview" else crop_fit(path, crop_box, source_size, cell)
            sheet.paste(tile, (col * cell[0], y))
            draw.rectangle((col * cell[0], y, col * cell[0] + cell[0] - 1, y + cell[1] - 1), outline=(220, 220, 220))
        label = (
            f"{key} | {original_method} -> {METHOD_LABELS.get(baseline, baseline)} | "
            f"target={row.get('target_tokens', '')} | host={row.get('source_tokens', '')}"
        )
        ly = y + cell[1] + 6
        for line in wrap(draw, label, label_font, sheet.width - 12)[:4]:
            draw.text((6, ly), line, fill=(25, 25, 25), font=label_font)
            ly += 18
        y += row_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-tree", type=Path, required=True)
    ap.add_argument("--probe-manifest", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v1_manifest.csv"))
    ap.add_argument("--fe-manifest", type=Path, default=Path("remote_patch/manifest_sam_135.json"))
    ap.add_argument("--audit", type=Path, default=Path("fe135_strict_review_20260626_seed10/strict_visual_audit_final.csv"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/fe135_t3_samflow_probe_v1"))
    ap.add_argument("--out", type=Path, default=Path("fe135_t3_samflow_probe_v1_review"))
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    probe_rows = list(csv.DictReader(args.probe_manifest.open(newline="", encoding="utf-8")))
    manifest = json.loads(args.fe_manifest.read_text(encoding="utf-8"))
    manifest_by_key = {item["key"]: item for item in manifest}
    audit = audit_by_pair(args.audit)
    args.out.mkdir(parents=True, exist_ok=True)

    status_path = args.out / "probe_file_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["baseline", "key", "current_exists", "probe_exists", "probe_path"])
        writer.writeheader()
        for row in probe_rows:
            original_method = "sd3" if row["baseline"] == "sam_flow_sd3" else "flux"
            key = row["task"]
            current = Path(audit[(original_method, key)]["result_path"])
            probe = map_project_path(args.project_tree, row.get("result_image", ""))
            if not probe:
                probe = args.project_tree / args.output_root / row["baseline"] / key / "seed_10" / "result.png"
            writer.writerow(
                {
                    "baseline": row["baseline"],
                    "key": key,
                    "current_exists": current.exists(),
                    "probe_exists": probe.exists(),
                    "probe_path": str(probe),
                }
            )

    chunks = [probe_rows[i : i + args.chunk] for i in range(0, len(probe_rows), args.chunk)]
    for index, chunk in enumerate(chunks):
        for mode in ("overview", "zoom"):
            make_sheet(
                out_path=args.out / f"{mode}_part{index + 1:02d}.jpg",
                rows=chunk,
                manifest_by_key=manifest_by_key,
                audit=audit,
                project_tree=args.project_tree,
                output_root=args.output_root,
                mode=mode,
                part_index=index,
                part_count=len(chunks),
            )
    missing = sum(1 for row in csv.DictReader(status_path.open(encoding="utf-8")) if row["probe_exists"] != "True")
    print(args.out)
    print("rows", len(probe_rows))
    print("missing_probe", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
