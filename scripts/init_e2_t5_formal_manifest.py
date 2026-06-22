#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = [
    "baseline",
    "task",
    "seed",
    "status",
    "source_image",
    "source_prompt",
    "target_prompt",
    "result_image",
    "metadata",
    "command",
    "matched_conditions",
    "failure_reason",
    "notes",
]

TASKS = {
    "pillow_same_color_cable_knit": {
        "source_image": "data/phase2_candidates/pexels_white_pillow_brown_sofa_6312089.jpg",
        "source_prompt": "A cozy living room photo with a plain white pillow on a brown sofa.",
        "target_prompt": "A photo of the same white pillow on the same brown sofa, with the entire pillow surface changed into same-color white chunky cable-knit fabric with thick braided knitted columns, while preserving the pillow shape, outline, lighting, sofa, background, and the rest of the scene.",
    },
    "pillow_same_color_linen_panel": {
        "source_image": "data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg",
        "source_prompt": "A cozy living room photo with a plain dark grey pillow on a rattan-backed sofa beside a wooden table, soft natural light, and a simple home interior.",
        "target_prompt": "A photo of the same plain dark grey pillow on the same rattan-backed sofa, with only the center surface panel changed into same-color dark grey woven linen fabric with fine natural cross-weave texture, while preserving the pillow shape, color, folds, lighting, sofa, rattan backrest, and background.",
    },
    "pillow_same_color_terry_panel": {
        "source_image": "data/phase2_candidates/pexels_green_chair_white_pillow_6312055.jpg",
        "source_prompt": "A photo of a plain white throw pillow centered on a green velvet armchair against a tiled wall.",
        "target_prompt": "A photo of the same plain white throw pillow on the same green velvet armchair, with only the center rectangular surface panel changed into same-color white terry cloth texture, while preserving the pillow shape, seams, shadows, chair, tiled wall, floor, and all surrounding pillow fabric.",
    },
    "pillow_same_color_cable_knit_grey": {
        "source_image": "data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg",
        "source_prompt": "A cozy living room photo with a plain dark grey pillow on a rattan-backed sofa beside a wooden table, soft natural light, and a simple home interior.",
        "target_prompt": "A photo of the same plain dark grey pillow on the same rattan-backed sofa, with the entire pillow surface uniformly changed into same-color dark grey chunky cable-knit fabric, the same thick braided knitted columns covering every part of the pillow including the brightly lit left side, while preserving the pillow shape, outline, lighting, sofa, table, wall, and the rest of the scene.",
    },
    "pillow_same_color_cable_knit_armchair": {
        "source_image": "data/phase2_candidates/pexels_green_chair_white_pillow_6312055.jpg",
        "source_prompt": "A photo of a plain white throw pillow centered on a green velvet armchair against a tiled wall.",
        "target_prompt": "A photo of the same plain white throw pillow on the same green velvet armchair, with the entire pillow surface changed into same-color white chunky cable-knit fabric with thick braided knitted columns, while preserving the pillow shape, seams, shadows, chair, tiled wall, floor, and the rest of the scene.",
    },
}

FORMAL_T5_TASKS = (
    "pillow_same_color_cable_knit",
    "pillow_same_color_cable_knit_grey",
    "pillow_same_color_cable_knit_armchair",
)

BASELINES = {
    "flowedit": ("E2.2 same-backbone SD3 target-mode RF baseline", "stabilityai/stable-diffusion-3-medium-diffusers"),
    "flowalign": ("E2.2 same-backbone SD3 target-mode RF baseline", "stabilityai/stable-diffusion-3-medium-diffusers"),
    "splitflow": ("E2.2 same-backbone SD3 target-mode RF baseline", "stabilityai/stable-diffusion-3-medium-diffusers + prompt decomposition"),
    "sam_flow_sd3": ("E2.2 same-backbone SD3 source-anchored masked-flow baseline", "stabilityai/stable-diffusion-3-medium-diffusers"),
    "fireflow": ("E2.3 native-FLUX contextual baseline", "black-forest-labs/FLUX.1-dev"),
    "rf_solver_edit": ("E2.3 native-FLUX contextual baseline", "black-forest-labs/FLUX.1-dev"),
    "reflex": ("E2.3 native-FLUX contextual baseline", "black-forest-labs/FLUX.1-dev"),
    "sam_flow_flux": ("E2.3 native-FLUX source-anchored masked-flow contextual baseline", "black-forest-labs/FLUX.1-dev"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/support_v3_2026-06-02/e2_t5_formal_baseline_manifest.csv"))
    parser.add_argument("--baselines", default=" ".join(BASELINES))
    parser.add_argument("--tasks", default=" ".join(FORMAL_T5_TASKS))
    parser.add_argument("--seeds", default="10 11 12")
    args = parser.parse_args()

    baselines = [item for item in args.baselines.split() if item]
    tasks = [item for item in args.tasks.split() if item]
    seeds = [item.removeprefix("seed_") for item in args.seeds.split() if item]
    missing_baselines = sorted(set(baselines) - set(BASELINES))
    missing_tasks = sorted(set(tasks) - set(TASKS))
    if missing_baselines or missing_tasks:
        raise SystemExit(f"Unknown baselines={missing_baselines} tasks={missing_tasks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for baseline in baselines:
            bucket, backbone = BASELINES[baseline]
            for task in tasks:
                spec = TASKS[task]
                for seed in seeds:
                    writer.writerow(
                        {
                            "baseline": baseline,
                            "task": task,
                            "seed": seed,
                            "status": "pending",
                            "source_image": spec["source_image"],
                            "source_prompt": spec["source_prompt"],
                            "target_prompt": spec["target_prompt"],
                            "result_image": "",
                            "metadata": "",
                            "command": "",
                            "matched_conditions": "",
                            "failure_reason": "",
                            "notes": f"{bucket}; {backbone}",
                        }
                    )
    print(f"Wrote {args.out}")
    print(f"rows={len(baselines) * len(tasks) * len(seeds)} baselines={','.join(baselines)} tasks={len(tasks)} seeds={','.join(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
