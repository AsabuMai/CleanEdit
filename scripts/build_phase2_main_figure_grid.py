from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
OUT_DIR = ROOT / "paper" / "assets"
OUT_IMAGE = OUT_DIR / "phase2_main_qual_grid_seed12_2026-06-11.png"
OUT_MANIFEST = OUT_DIR / "phase2_main_qual_grid_seed12_2026-06-11.json"

SOURCE_FILES = [
    EXP / "e1_t1_t4_directgeneric_metrics.csv",
    EXP / "table2a_e4_common_subset_clipdino_metrics.csv",
    EXP / "table2_t5_internal_metrics.csv",
]
FINAL_REGISTRY = EXP / "phase2_final_selected_runs_2026-06-11.csv"

TASKS = [
    ("T1 attached accessory", "cat_crown", "Cat crown"),
    ("T2 container insertion", "bowl_apple_inside", "Bowl + apple"),
    ("T3 surface decal", "tshirt_star", "T-shirt star"),
    ("T4 local recolor", "red_office_chair_to_blue_office_chair", "Office chair recolor"),
    ("T5 same-color material", "pillow_same_color_cable_knit", "Cable-knit pillow"),
]
METHODS = [
    ("direct_target", "Direct target"),
    ("adaptive_full_generic_support", "Generic support"),
    ("support_v3_controller_rmsgap", "CleanEdit"),
]
SEED = "12"
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in SOURCE_FILES:
        for row in read_rows(path):
            task = row.get("task", "")
            method = row.get("method", "")
            seed = row.get("seed", "").removeprefix("seed_")
            if not task or not method or not seed:
                continue
            rows[(task, seed, method)] = row
    return rows


def load_final_registry() -> dict[tuple[str, str, str], dict[str, str]]:
    registry: dict[tuple[str, str, str], dict[str, str]] = {}
    if not FINAL_REGISTRY.exists():
        return registry
    for row in read_rows(FINAL_REGISTRY):
        task = row.get("task", "")
        seed = row.get("seed", "")
        method = row.get("method", "")
        if task and seed and method:
            registry[(task, seed, method)] = row
    return registry


def resolve_path(path_value: str) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    candidate = ROOT / path_value
    if candidate.exists():
        return candidate
    return None


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), RESAMPLE)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_width: int, fill: tuple[int, int, int], fnt: ImageFont.ImageFont) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=fnt)
        y += int(fnt.size * 1.25)


def main() -> int:
    rows = load_rows()
    registry = load_final_registry()
    missing: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    thumb = 236
    left_w = 260
    pad = 18
    header_h = 76
    row_h = thumb + 18
    columns = [("source", "Source"), *METHODS]
    width = left_w + pad + len(columns) * (thumb + pad)
    height = header_h + pad + len(TASKS) * (row_h + pad)

    grid = Image.new("RGB", (width, height), (248, 248, 247))
    draw = ImageDraw.Draw(grid)
    header_font = font(22, bold=True)
    task_font = font(20, bold=True)
    small_font = font(16)

    draw.text((pad, 22), "Phase2 T1-T5", fill=(15, 15, 15), font=header_font)
    for col_index, (_, label) in enumerate(columns):
        x = left_w + pad + col_index * (thumb + pad)
        draw.text((x, 22), label, fill=(15, 15, 15), font=header_font)

    for row_index, (family, task, short_label) in enumerate(TASKS):
        y = header_h + pad + row_index * (row_h + pad)
        draw_wrapped(draw, f"{family}: {short_label}", (pad, y + 70), left_w - 2 * pad, (20, 20, 20), task_font)

        source_row = rows.get((task, SEED, METHODS[0][0])) or rows.get((task, SEED, METHODS[-1][0]))
        source_reg = registry.get((task, SEED, METHODS[0][0])) or registry.get((task, SEED, METHODS[-1][0]))
        source_path = resolve_path(source_reg.get("selected_source_image", "")) if source_reg else None
        if not source_path and source_row:
            source_path = resolve_path(source_row.get("source_image", ""))
        if not source_path:
            missing.append({"task": task, "seed": SEED, "method": "source"})
            continue

        cells = [fit(source_path, thumb)]
        manifest_rows.append(
            {
                "family": family,
                "task": task,
                "seed": SEED,
                "column": "Source",
                "method": "source",
                "image": str(source_path),
            }
        )

        for method, label in METHODS:
            row = rows.get((task, SEED, method))
            reg_row = registry.get((task, SEED, method))
            result_path = resolve_path(reg_row.get("selected_result_image", "")) if reg_row else None
            if not result_path and row:
                result_path = resolve_path(row.get("result_image", ""))
            if not result_path:
                missing.append({"task": task, "seed": SEED, "method": method})
                cells.append(Image.new("RGB", (thumb, thumb), (230, 230, 230)))
                continue
            cells.append(fit(result_path, thumb))
            manifest_rows.append(
                {
                    "family": family,
                    "task": task,
                    "seed": SEED,
                    "column": label,
                    "method": method,
                    "image": str(result_path),
                    "image_source": "phase2_final_selected_runs_2026-06-11.csv" if reg_row else "metric_row",
                    "selected_run": reg_row.get("selected_run", "") if reg_row else "",
                    "selection_reason": reg_row.get("selection_reason", "") if reg_row else "",
                    "metric_status": reg_row.get("metric_status", "") if reg_row else "",
                    "outside_mask_l1": row.get("outside_mask_l1", "") if row else "",
                    "source_ssim_luma": row.get("source_ssim_luma", "") if row else "",
                    "edit_score": row.get("edit_score", "") if row else "",
                }
            )

        for col_index, cell in enumerate(cells):
            x = left_w + pad + col_index * (thumb + pad)
            grid.paste(cell, (x, y))
            draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), outline=(218, 218, 218), width=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.save(OUT_IMAGE)
    manifest = {
        "status": "complete" if not missing else "incomplete",
        "scope": "Main qualitative grid for Phase2 closure; one representative seed-12 case per T1-T5 family.",
        "output_image": str(OUT_IMAGE.relative_to(ROOT)),
        "seed": SEED,
        "registry": str(FINAL_REGISTRY.relative_to(ROOT)) if FINAL_REGISTRY.exists() else "",
        "registry_rows_loaded": len(registry),
        "columns": [label for _, label in columns],
        "rows": manifest_rows,
        "missing": missing,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"status={manifest['status']}")
    print(f"image={OUT_IMAGE.relative_to(ROOT)}")
    print(f"manifest={OUT_MANIFEST.relative_to(ROOT)}")
    if missing:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
