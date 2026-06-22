from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REMOTE_PROJECT = Path("/cluster/users/grad/2025/25t8103/project")
if REMOTE_PROJECT.exists():
    PROJECT = REMOTE_PROJECT
    ROOT = PROJECT
    DIR = PROJECT / "experiments" / "support_v3_2026-06-02"
    OUT_DIR = PROJECT / "paper" / "assets" / "appendix_grids"
else:
    ROOT = Path(r"I:\Downloads\1\2")
    PROJECT = ROOT
    DIR = ROOT / "phase2_lock_2026-06-11"
    OUT_DIR = DIR / "appendix_grids"

FILES = [
    DIR / "strict_fixed_mask_metrics.csv",
    DIR / "e1_t1_t4_directgeneric_metrics.csv",
    DIR / "table2a_e4_common_subset_clipdino_metrics.csv",
    DIR / "e2_t1_t4_baseline_fixed_mask_metrics.csv",
    DIR / "table2_t5_internal_metrics.csv",
    DIR / "table2_t5_baseline_metrics.csv",
]
FINAL_REGISTRY = DIR / "phase2_final_selected_runs_2026-06-11.csv"

FAMILIES = [
    (
        "T1_attached_accessory",
        "T1 Attached accessory",
        [
            ("cat_crown", "Cat crown"),
            ("dog_bow_tie_phase2", "Dog bow tie"),
            ("dog_front_sunglasses_phase2", "Dog sunglasses"),
        ],
    ),
    (
        "T2_container_insertion",
        "T2 Container insertion",
        [
            ("bowl_apple_inside", "Bowl + apple"),
            ("white_bowl_orange_tabletop_phase2", "White bowl + orange"),
            ("brown_bowl_lemon_phase2", "Brown bowl + lemon"),
        ],
    ),
    (
        "T3_surface_decal",
        "T3 Surface decal",
        [
            ("tshirt_star", "T-shirt star"),
            ("mug_heart", "Mug heart"),
            ("tote_leaf", "Tote leaf"),
        ],
    ),
    (
        "T4_local_recolor",
        "T4 Local recolor",
        [
            ("red_office_chair_to_blue_office_chair", "Office chair blue"),
            ("green_mug_orange_phase2", "Green mug orange"),
            ("yellow_vase_blue_phase2", "Yellow vase blue"),
        ],
    ),
    (
        "T5_same_color_material",
        "T5 Same-color material",
        [
            ("pillow_same_color_cable_knit", "White cable-knit pillow"),
            ("pillow_same_color_cable_knit_grey", "Grey cable-knit pillow"),
            ("pillow_same_color_cable_knit_armchair", "Armchair cable-knit pillow"),
        ],
    ),
]

METHODS = [
    ("base_only", "RF recon"),
    ("direct_target", "Direct"),
    ("adaptive_full_generic_support", "Generic"),
    ("flowedit", "FlowEdit"),
    ("splitflow", "SplitFlow"),
    ("sam_flow_sd3", "Sam-Flow SD3"),
    ("fireflow", "FireFlow"),
    ("rf_solver_edit", "RF-Solver"),
    ("reflex", "ReFlex"),
    ("sam_flow_flux", "Sam-Flow FLUX"),
    ("support_v3_fixed", "Fixed DeCE"),
    ("support_v3_controller_rmsgap", "DeCE-RF"),
]

SEED = "12"
TASK_SEED_OVERRIDES: dict[str, str] = {}
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def load_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in FILES:
        for row in read_csv(path):
            task = row.get("task", "")
            seed = row.get("seed", "").removeprefix("seed_")
            method = row.get("method", "")
            if not task or not seed or not method:
                continue
            key = (task, seed, method)
            rows.setdefault(key, row)
    return rows


def load_final_registry() -> dict[tuple[str, str, str], dict[str, str]]:
    registry: dict[tuple[str, str, str], dict[str, str]] = {}
    if not FINAL_REGISTRY.exists():
        return registry
    for row in read_csv(FINAL_REGISTRY):
        task = row.get("task", "")
        seed = row.get("seed", "")
        method = row.get("method", "")
        if task and seed and method:
            registry[(task, seed, method)] = row
    return registry


def load_source_manifest() -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in [
        PROJECT / "data" / "phase2_candidates" / "sources_manifest.csv",
        DIR / "normalized_512" / "t1_t4_eval_assets_manifest.csv",
    ]:
        if not path.exists():
            continue
        for row in read_csv(path):
            task = row.get("task", "")
            image = row.get("source_image", "") or row.get("source", "")
            if task and image:
                sources[task] = image
    return sources


def resolve_image(path_value: str) -> Path | None:
    p = Path(path_value)
    if p.exists():
        return p
    if p.is_absolute():
        # Map the cluster project root into the local downloaded bundle when possible.
        parts = p.as_posix().split("/project/", 1)
        if len(parts) == 2:
            candidate = DIR / "remote_images" / parts[1].replace("/", "__")
            if candidate.exists():
                return candidate
        return None
    candidate = ROOT / path_value
    if candidate.exists():
        return candidate
    candidate = PROJECT / path_value
    if candidate.exists():
        return candidate
    return None


def task_seed(task: str) -> str:
    return TASK_SEED_OVERRIDES.get(task, SEED)


def normalized_source(task: str) -> Path | None:
    candidate = DIR / "normalized_512" / "sources" / f"{task}.png"
    return candidate if candidate.exists() else None


def source_for_task(task: str, sources: dict[str, str]) -> Path | None:
    if src := normalized_source(task):
        return src
    if path_value := sources.get(task, ""):
        return resolve_image(path_value)
    return None


def registry_result(
    registry: dict[tuple[str, str, str], dict[str, str]], task: str, seed: str, method: str
) -> tuple[Path | None, dict[str, str] | None]:
    row = registry.get((task, seed, method))
    if not row:
        return None, None
    return resolve_image(row.get("selected_result_image", "")), row


def fallback_result(task: str, seed: str, method: str) -> Path | None:
    candidates = [
        PROJECT / "outputs" / "pretty_matrix" / task / method / f"seed_{seed}" / "result.png",
        PROJECT / "outputs" / "e2_t1_t4_baseline_matrix" / task / method / f"seed_{seed}" / "result.png",
        PROJECT / "outputs" / "baselines" / method / task / f"seed_{seed}" / "result.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def copy_remote_images(rows: dict[tuple[str, str, str], dict[str, str]]) -> None:
    """This script expects images to already be available locally or downloaded separately."""
    (DIR / "remote_images").mkdir(exist_ok=True)


def fit(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), RESAMPLE)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def placeholder(size: int, text: str) -> Image.Image:
    img = Image.new("RGB", (size, size), (238, 238, 238))
    draw = ImageDraw.Draw(img)
    draw.text((14, size // 2 - 16), text, fill=(90, 90, 90), font=font(16, True))
    return img


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, max_width: int, fnt: ImageFont.ImageFont) -> None:
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines[:3]:
        draw.text((x, y), line, fill=(35, 35, 35), font=fnt)
        y += int(fnt.size * 1.25)


def make_family_grid(
    rows: dict[tuple[str, str, str], dict[str, str]],
    sources: dict[str, str],
    registry: dict[tuple[str, str, str], dict[str, str]],
    family_id: str,
    family_label: str,
    tasks: list[tuple[str, str]],
) -> dict[str, object]:
    thumb = 124
    left_w = 235
    pad = 10
    header_h = 92
    row_h = thumb + 22
    columns = [("source", "Source"), *METHODS]
    width = left_w + pad + len(columns) * (thumb + pad)
    height = header_h + pad + len(tasks) * (row_h + pad)
    img = Image.new("RGB", (width, height), (248, 248, 247))
    draw = ImageDraw.Draw(img)
    draw.text((22, 22), f"Appendix {family_label}", fill=(20, 38, 55), font=font(28, True))
    draw.text((22, 58), f"Seed {SEED}; source plus formal main/external baseline methods", fill=(90, 90, 90), font=font(16))

    for ci, (_, label) in enumerate(columns):
        x = left_w + pad + ci * (thumb + pad)
        draw_label(draw, (x, 34), label, thumb, font(13, True))

    missing: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    for ri, (task, task_label) in enumerate(tasks):
        seed = task_seed(task)
        y = header_h + pad + ri * (row_h + pad)
        label = task_label if seed == SEED else f"{task_label} (seed {seed})"
        draw_label(draw, (22, y + 38), label, left_w - 34, font(16, True))

        cells: list[Image.Image] = []
        if src := source_for_task(task, sources):
            cells.append(fit(src, thumb))
            manifest_rows.append({"task": task, "seed": seed, "method": "source", "image": str(src)})
        else:
            cells.append(placeholder(thumb, "missing source"))
            missing.append({"task": task, "seed": seed, "method": "source"})

        for method, label in METHODS:
            row = rows.get((task, seed, method))
            reg_result, reg_row = registry_result(registry, task, seed, method)
            if reg_result:
                cells.append(fit(reg_result, thumb))
                manifest_rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "method": method,
                        "image": str(reg_result),
                        "source": "phase2_final_selected_runs_2026-06-11.csv",
                        "selected_run": reg_row.get("selected_run", "") if reg_row else "",
                        "selection_reason": reg_row.get("selection_reason", "") if reg_row else "",
                        "metric_status": reg_row.get("metric_status", "") if reg_row else "",
                    }
                )
            elif row and (result := resolve_image(row.get("result_image", ""))):
                cells.append(fit(result, thumb))
                manifest_rows.append({"task": task, "seed": seed, "method": method, "image": str(result)})
            elif result := fallback_result(task, seed, method):
                cells.append(fit(result, thumb))
                manifest_rows.append({"task": task, "seed": seed, "method": method, "image": str(result), "source": "fallback"})
            else:
                cells.append(placeholder(thumb, "missing"))
                missing.append({"task": task, "seed": seed, "method": method})

        for ci, cell in enumerate(cells):
            x = left_w + pad + ci * (thumb + pad)
            img.paste(cell, (x, y))
            draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), outline=(218, 218, 218), width=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"appendix_{family_id}_representative_all_methods.png"
    img.save(out)
    return {
        "family": family_id,
        "label": family_label,
        "output": str(out),
        "missing": missing,
        "rows": manifest_rows,
    }


def main() -> None:
    rows = load_rows()
    sources = load_source_manifest()
    registry = load_final_registry()
    reports = [make_family_grid(rows, sources, registry, family_id, label, tasks) for family_id, label, tasks in FAMILIES]
    manifest = {
        "status": "complete" if not any(r["missing"] for r in reports) else "incomplete",
        "scope": "Appendix visual comparison grids, source plus formal main/external Phase2 methods. Seed 12 is used for every Phase2 task case.",
        "methods": [label for _, label in METHODS],
        "seed": SEED,
        "registry": str(FINAL_REGISTRY),
        "registry_rows_loaded": len(registry),
        "task_seed_overrides": TASK_SEED_OVERRIDES,
        "families": reports,
    }
    path = OUT_DIR / "appendix_phase2_all_methods_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"status={manifest['status']}")
    print(path)
    for report in reports:
        print(report["output"], "missing", len(report["missing"]))
    if manifest["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
