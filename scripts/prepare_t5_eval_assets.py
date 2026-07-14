from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path.cwd()
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
MATRIX = ROOT / "outputs" / "pretty_matrix"
CANVAS = 512
SEEDS = ("10", "11", "12")

TASKS = {
    "pillow_same_color_cable_knit": "data/phase2_candidates/pexels_white_pillow_brown_sofa_6312089.jpg",
    "pillow_same_color_cable_knit_grey": "data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg",
    "pillow_same_color_cable_knit_armchair": "data/phase2_candidates/pexels_green_chair_white_pillow_6312055.jpg",
}


def fit_to_canvas(image: Image.Image, mode: str) -> Image.Image:
    image = image.convert(mode)
    scale = min(CANVAS / image.width, CANVAS / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.Resampling.BICUBIC if mode == "RGB" else Image.Resampling.BILINEAR
    fitted = image.resize(size, resample)
    fill = (245, 245, 245) if mode == "RGB" else 0
    canvas = Image.new(mode, (CANVAS, CANVAS), fill)
    canvas.paste(fitted, ((CANVAS - size[0]) // 2, (CANVAS - size[1]) // 2))
    return canvas


def load_mask(path: Path) -> Image.Image:
    return fit_to_canvas(Image.open(path).convert("L"), "L")


def aggregate_eval_mask(task: str) -> tuple[Image.Image, list[str]]:
    paths = []
    for seed in SEEDS:
        path = MATRIX / task / "support_v3_controller_rmsgap" / f"seed_{seed}" / "masks" / "operation_v3_edit_mask.png"
        if path.is_file():
            paths.append(path)
    if len(paths) != len(SEEDS):
        raise FileNotFoundError(f"{task}: expected {len(SEEDS)} CleanEdit support masks, found {len(paths)}")
    masks = [np.asarray(load_mask(path), dtype=np.float32) / 255.0 for path in paths]
    mean_mask = np.mean(np.stack(masks, axis=0), axis=0)
    binary = (mean_mask >= 0.35).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L"), [str(path.relative_to(ROOT)) for path in paths]


def main() -> int:
    source_dir = EXP / "normalized_512" / "sources"
    mask_dir = EXP / "normalized_512" / "eval_masks"
    source_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, rel_source in TASKS.items():
        source_path = ROOT / rel_source
        source_out = source_dir / f"{task}.png"
        mask_out = mask_dir / f"{task}_eval_mask.png"
        fit_to_canvas(Image.open(source_path), "RGB").save(source_out)
        mask, mask_sources = aggregate_eval_mask(task)
        mask.save(mask_out)
        rows.append(
            {
                "task": task,
                "source_image": rel_source,
                "normalized_source": str(source_out.relative_to(ROOT)),
                "eval_mask": str(mask_out.relative_to(ROOT)),
                "eval_mask_source": "mean support_v3_controller_rmsgap operation_v3_edit_mask over seeds 10/11/12, threshold 0.35",
                "mask_inputs": ";".join(mask_sources),
            }
        )
    manifest = EXP / "normalized_512" / "t5_eval_assets_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "scope": "Phase2 T5 eval assets: three T5-1-style full-pillow cable-knit tasks",
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "canvas": CANVAS,
        "eval_mask_policy": "seed-aggregated CleanEdit operation support proxy over the full-pillow T5-1-style tasks; use for Table 2 only after visual acceptance",
        "manifest": str(manifest.relative_to(ROOT)),
    }
    (EXP / "normalized_512" / "t5_eval_assets_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest}")
    print(f"tasks={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
