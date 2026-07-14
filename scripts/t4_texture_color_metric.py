from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
COLOR_RGB = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "red": (200, 30, 30),
    "orange": (230, 130, 20),
    "yellow": (235, 215, 40),
    "gold": (212, 175, 55),
    "golden": (212, 175, 55),
    "green": (40, 160, 60),
    "blue": (40, 90, 200),
    "cyan": (40, 200, 210),
    "teal": (30, 150, 150),
    "purple": (130, 50, 160),
    "violet": (140, 80, 200),
    "pink": (235, 130, 180),
    "brown": (110, 70, 40),
    "beige": (220, 200, 160),
    "silver": (190, 190, 195),
}
COLORS = tuple(COLOR_RGB)


def colors_in(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return [c for c in COLORS if re.search(r"\b" + re.escape(c) + r"\b", text)]


def src_tgt(entry: dict) -> tuple[str | None, str | None]:
    key = entry.get("key", "")
    if key == "fe_199_piece_of_cake_2_red_velvet_cake":
        return "brown", "red"
    src = colors_in(entry["source_prompt"])
    tgt = colors_in(entry["target_prompt"])
    source = src[0] if src else None
    target = None
    for color in tgt:
        if color not in set(src):
            target = color
            break
    return source, target or (tgt[-1] if tgt else None)


def lab_of(name: str) -> np.ndarray:
    rgb = np.array(COLOR_RGB[name], dtype=float).reshape(1, 1, 3) / 255.0
    return rgb2lab(rgb)[0, 0]


def mean(values: list[float]) -> float | str:
    return "" if not values else float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=PROJ / "outputs/t4_texture_metric_runs")
    parser.add_argument("--manifest", type=Path, default=PROJ / "data/flowedit_compatible_135/manifest_t4_recolor_19.json")
    parser.add_argument("--eval-mask-dir", type=Path, default=PROJ / "data/flowedit_compatible_135/eval_masks")
    parser.add_argument("--csv-output", type=Path, default=PROJ / "experiments/t4_texture_metrics/color_metrics.csv")
    parser.add_argument("--summary-output", type=Path, default=PROJ / "experiments/t4_texture_metrics/color_summary.csv")
    parser.add_argument(
        "--paper-strict",
        action="store_true",
        help="Use the paper's 16 single-color T4 cases: exclude boat hull, colorful duck, and red-flower.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    methods = sorted(p.name for p in next(args.runs.iterdir()).iterdir() if p.is_dir())
    rows = []
    accum = {m: {"toward_target": [], "de_improve": [], "n": 0} for m in methods}
    for entry in manifest:
        key = entry["key"]
        if args.paper_strict and key in {
            "fe_017_boat_silhouette_1_sailboat_white_sail_red_hull",
            "fe_029_butterfly_2_red_flower",
            "fe_108_duck_1_colorful_duck",
        }:
            continue
        source_name, target_name = src_tgt(entry)
        if source_name is None or target_name is None:
            continue
        source = Image.open(entry["image"]).convert("RGB")
        width, height = source.size
        mask_path = args.eval_mask_dir / f"{key}_eval_mask.png"
        if not mask_path.exists():
            continue
        region = np.asarray(Image.open(mask_path).convert("L").resize((width, height))) > 127
        if int(region.sum()) < 50:
            continue
        source_lab = rgb2lab(np.asarray(source, dtype=float) / 255.0)
        source_anchor = lab_of(source_name)
        target_anchor = lab_of(target_name)
        axis = target_anchor - source_anchor
        axis_unit = axis / (np.linalg.norm(axis) + 1e-6)
        de_source = np.linalg.norm(source_lab[region] - target_anchor, axis=1).mean()
        for method in methods:
            result_path = args.runs / key / method / "seed_10/result.png"
            if not result_path.exists():
                continue
            result = Image.open(result_path).convert("RGB").resize((width, height))
            result_lab = rgb2lab(np.asarray(result, dtype=float) / 255.0)
            delta = result_lab[region] - source_lab[region]
            toward = float((delta @ axis_unit).mean())
            de_result = np.linalg.norm(result_lab[region] - target_anchor, axis=1).mean()
            de_improve = float(de_source - de_result)
            rows.append(
                {
                    "task": key,
                    "method": method,
                    "source_color": source_name,
                    "target_color": target_name,
                    "toward_target": toward,
                    "de_improve": de_improve,
                }
            )
            accum[method]["toward_target"].append(toward)
            accum[method]["de_improve"].append(de_improve)
            accum[method]["n"] += 1

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "method", "source_color", "target_color", "toward_target", "de_improve"])
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for method, data in sorted(accum.items()):
        summary.append(
            {
                "method": method,
                "n": data["n"],
                "S_color": mean(data["toward_target"]),
                "dE_improve": mean(data["de_improve"]),
            }
        )
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "n", "S_color", "dE_improve"])
        writer.writeheader()
        writer.writerows(summary)
    print("color csv", args.csv_output)
    print("color summary", args.summary_output)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
