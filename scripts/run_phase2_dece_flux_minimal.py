#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
from pathlib import Path


T1_T4_MANIFEST = "experiments/support_v3_2026-06-02/e2_t1_t4_formal_baseline_manifest.csv"
T5_MANIFEST = "experiments/support_v3_2026-06-02/e2_t5_formal_baseline_manifest.csv"

TASKS = [
    "cat_crown",
    "dog_bow_tie_phase2",
    "dog_front_sunglasses_phase2",
    "bowl_apple_inside",
    "white_bowl_orange_tabletop_phase2",
    "brown_bowl_lemon_phase2",
    "tshirt_star",
    "mug_heart",
    "tote_leaf",
    "red_office_chair_to_blue_office_chair",
    "green_mug_orange_phase2",
    "yellow_vase_blue_phase2",
    "pillow_same_color_cable_knit",
    "pillow_same_color_cable_knit_grey",
    "pillow_same_color_cable_knit_armchair",
]

TASK_CFG = {
    "cat_crown": {"op": "add_object", "relation": "above_host", "edit_scale": 1.35, "overlay": "crown"},
    "dog_bow_tie_phase2": {"op": "add_object", "relation": "below_host", "edit_scale": 1.35},
    "dog_front_sunglasses_phase2": {"op": "add_object", "relation": "face_accessory", "edit_scale": 1.35},
    "bowl_apple_inside": {"op": "add_object", "relation": "inside", "edit_scale": 1.25},
    "white_bowl_orange_tabletop_phase2": {"op": "add_object", "relation": "surface", "edit_scale": 1.75, "overlay": "orange"},
    "brown_bowl_lemon_phase2": {"op": "add_object", "relation": "inside", "edit_scale": 1.25},
    "tshirt_star": {"op": "surface_decal", "relation": "surface", "edit_scale": 1.45, "overlay": "star"},
    "mug_heart": {"op": "surface_decal", "relation": "surface", "edit_scale": 1.45, "overlay": "heart"},
    "tote_leaf": {"op": "surface_decal", "relation": "surface", "edit_scale": 1.45, "overlay": "leaf"},
    "red_office_chair_to_blue_office_chair": {"op": "recolor", "relation": "inside", "color": "blue", "source_color": "red", "edit_scale": 1.0},
    "green_mug_orange_phase2": {"op": "recolor", "relation": "inside", "color": "orange", "source_color": "green", "edit_scale": 1.0},
    "yellow_vase_blue_phase2": {"op": "recolor", "relation": "inside", "color": "blue", "source_color": "yellow", "edit_scale": 1.0},
    "pillow_same_color_cable_knit": {"op": "material", "relation": "inside", "edit_scale": 2.0, "source_color": "light_neutral"},
    "pillow_same_color_cable_knit_grey": {"op": "material", "relation": "inside", "edit_scale": 2.0, "source_color": "grey"},
    "pillow_same_color_cable_knit_armchair": {"op": "material", "relation": "inside", "edit_scale": 2.0, "source_color": "light_neutral"},
}


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("seed", "")).removeprefix("seed_") == "10":
                rows.setdefault(row["task"], row)
    return rows


def command_for(
    *,
    project: Path,
    row: dict[str, str],
    task: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cfg = TASK_CFG[task]
    mask = project / "experiments" / "support_v3_2026-06-02" / "normalized_512" / "eval_masks" / f"{task}_eval_mask.png"
    cmd = [
        str(project / ".venv" / "bin" / "python"),
        str(project / "run_edit_flux.py"),
        "--model-id",
        args.model_id,
        "--cache-dir",
        str(project / ".cache" / "huggingface" / "hub"),
        "--local-files-only",
        "--model-offload",
        "--image",
        str(project / row["source_image"]),
        "--source-prompt",
        row["source_prompt"],
        "--prompt",
        row["target_prompt"],
        "--method",
        "dece_rf_flux_minimal",
        "--seed",
        "10",
        "--num-inference-steps",
        str(args.steps),
        "--n-max",
        str(args.n_max),
        "--max-image-size",
        str(args.max_image_size),
        "--src-guidance-scale",
        str(args.src_guidance),
        "--tar-guidance-scale",
        str(args.tar_guidance),
        "--base-guidance-scale",
        str(args.src_guidance),
        "--semantic-base-mask",
        str(mask),
        "--support-control-mode",
        "fixed",
        "--support-external-mask-role",
        "attention",
        "--edit-operation",
        cfg["op"],
        "--support-relation",
        cfg["relation"],
        "--mask-layering-mode",
        args.mask_layering_mode,
        "--minimal-core-scale",
        str(args.core_scale),
        "--minimal-ring-scale",
        str(args.ring_scale),
        "--minimal-outside-lock-scale",
        str(args.outside_lock),
        "--minimal-n-avg",
        str(args.n_avg),
        "--minimal-edit-scale",
        str(cfg.get("edit_scale", args.edit_scale)),
        "--final-postprocess-mode",
        "debug_visual" if cfg.get("overlay") or cfg["op"] == "material" else "mask_blend",
        "--output",
        str(out_dir / "result.png"),
        "--metadata-output",
        str(out_dir / "metadata.json"),
        "--stats-output",
        str(out_dir / "stats.json"),
    ]
    if cfg["op"] == "recolor":
        cmd.extend(
            [
                "--recolor-target",
                cfg["color"],
                "--recolor-clean-projection-scale",
                "1.0",
                "--minimal-recolor-scale",
                str(args.recolor_scale),
                "--final-recolor-blend-scale",
                str(args.final_recolor_blend),
                "--final-mask-dilate",
                str(args.recolor_mask_dilate),
                "--final-source-color-mask",
                cfg["source_color"],
            ]
        )
    if cfg["op"] == "material":
        cmd.extend(
            [
                "--final-knit-texture-scale",
                str(args.final_knit_texture),
                "--final-knit-source-blend",
                str(args.final_knit_source_blend),
                "--final-mask-dilate",
                str(args.material_mask_dilate),
                "--final-source-color-mask",
                cfg.get("source_color", "light_neutral"),
            ]
        )
    if cfg.get("overlay"):
        cmd.extend(["--final-object-overlay", str(cfg["overlay"])])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--tasks", default=" ".join(TASKS))
    parser.add_argument("--output-root", default="outputs/phase2_dece_flux_minimal_seed10_v1")
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--n-max", type=int, default=26)
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--src-guidance", type=float, default=1.5)
    parser.add_argument("--tar-guidance", type=float, default=10.5)
    parser.add_argument("--core-scale", type=float, default=1.0)
    parser.add_argument("--ring-scale", type=float, default=0.35)
    parser.add_argument("--outside-lock", type=float, default=1.0)
    parser.add_argument("--recolor-scale", type=float, default=0.25)
    parser.add_argument("--n-avg", type=int, default=4)
    parser.add_argument("--edit-scale", type=float, default=1.0)
    parser.add_argument("--final-recolor-blend", type=float, default=0.90)
    parser.add_argument("--final-knit-texture", type=float, default=2.4)
    parser.add_argument("--final-knit-source-blend", type=float, default=0.78)
    parser.add_argument("--recolor-mask-dilate", type=int, default=0)
    parser.add_argument("--material-mask-dilate", type=int, default=1)
    parser.add_argument("--mask-layering-mode", choices=["none", "object_contact"], default="none")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    host = socket.gethostname()
    if not args.dry_run and host != "a100-01.gpu01.cis.k.hosei.ac.jp":
        raise SystemExit(f"refusing heavy FLUX run on {host}")

    rows = {}
    rows.update(read_manifest(project / T1_T4_MANIFEST))
    rows.update(read_manifest(project / T5_MANIFEST))

    selected = [item for item in args.tasks.split() if item]
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    env = {
        **os.environ,
        "HF_HOME": str(project / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(project / ".cache" / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(project / ".cache" / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    }
    for task in selected:
        if task not in rows:
            raise SystemExit(f"missing seed10 manifest row for {task}")
        out_dir = output_root / task / "dece_rf_flux_minimal" / "seed_10"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = command_for(project=project, row=rows[task], task=task, out_dir=out_dir, args=args)
        (out_dir / "command.txt").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
        if args.skip_existing and (out_dir / "result.png").is_file() and (out_dir / "metadata.json").is_file():
            status = "skipped"
        elif args.dry_run:
            status = "dry_run"
        else:
            with (out_dir / "run.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            status = "complete" if proc.returncode == 0 else f"failed:{proc.returncode}"
            if proc.returncode != 0:
                summary.append({"task": task, "status": status, "out_dir": str(out_dir)})
                break
        summary.append({"task": task, "status": status, "out_dir": str(out_dir)})

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    for item in summary:
        print(item["task"], item["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
