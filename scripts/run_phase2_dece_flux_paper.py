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
SD3_SUPPORT_ROOT = "outputs/pretty_matrix"

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
    "cat_crown": {
        "op": "add_object",
        "relation": "above_host",
        "kind": "add",
        "overlay": "crown",
        "local_prompt": "One clearly visible small golden crown centered on top of a black cat's head between the ears.",
        "new_tokens": "crown golden",
        "host_tokens": "cat head ears",
        "support_mask": "outputs/phase2_dece_flux_custom_masks_20260616/cat_crown_big.png",
        "support_control_mode": "fixed",
        "params": {
            "local_target": 1.00,
            "edit_hedit": 1.10,
            "edit_anchor": 0.20,
            "edit_region": 0.34,
            "edit_target": 0.22,
            "region_transport": 0.28,
            "outside_lock": 0.05,
            "rec": 0.22,
            "struct": 0.45,
            "traj": 0.14,
            "preserve_gain": 2.5,
            "final_mask_alpha_gamma": 0.55,
        },
    },
    "dog_bow_tie_phase2": {
        "op": "add_object",
        "relation": "below_host",
        "kind": "add",
        "overlay": "bow_tie",
        "local_prompt": "One small red bow tie attached at the front of the dog's neck below the chin, with the dog's face unchanged.",
        "new_tokens": "bow tie",
        "host_tokens": "dog chin neck",
        "support_mask": "outputs/phase2_dece_flux_custom_masks_20260616/dog_bow_tie_wide.png",
        "support_control_mode": "fixed",
        "params": {
            "local_target": 1.00,
            "edit_hedit": 1.00,
            "edit_anchor": 0.25,
            "edit_region": 0.00,
            "edit_target": 0.18,
            "region_transport": 0.00,
            "rec": 0.30,
            "struct": 0.55,
            "traj": 0.20,
            "outside_lock": 0.25,
            "preserve_gain": 3.5,
            "core_attention_percentile": 25.0,
        },
    },
    "dog_front_sunglasses_phase2": {
        "op": "add_object",
        "relation": "on_face",
        "kind": "add",
        "overlay": "sunglasses",
        "target_prompt": "A close-up front-facing portrait of the same dog wearing black sunglasses aligned across both eyes indoors, while the dog face, ears, fur, nose, floor, and lighting remain unchanged.",
        "local_prompt": "One pair of black sunglasses aligned across both eyes of the same front-facing dog, with the nose, ears, fur, floor, and lighting unchanged.",
        "new_tokens": "sunglasses glasses",
        "host_tokens": "dog eyes face",
        "support_mask": "outputs/phase2_dece_flux_custom_masks_20260616/dog_sunglasses_eye_band.png",
        "support_control_mode": "fixed",
        "params": {
            "edit_hedit": 1.00,
            "edit_anchor": 0.22,
            "edit_region": 0.38,
            "edit_target": 0.18,
            "local_target": 1.00,
            "rec": 0.30,
            "struct": 0.55,
            "traj": 0.22,
            "region_transport": 0.28,
            "outside_lock": 0.35,
            "preserve_budget": 0.16,
            "preserve_gain": 4.0,
            "core_attention_percentile": 20.0,
            "final_mask_alpha_gamma": 1.0,
        },
    },
    "bowl_apple_inside": {
        "op": "add_object",
        "relation": "inside_container",
        "kind": "add",
        "local_prompt": "One single small red apple centered inside a blue ceramic bowl.",
        "new_tokens": "apple red",
        "host_tokens": "bowl inside",
    },
    "white_bowl_orange_tabletop_phase2": {
        "op": "add_object",
        "relation": "on_surface",
        "kind": "add",
        "overlay": "orange",
        "local_prompt": "One single orange sitting on the tabletop next to a white bowl.",
        "new_tokens": "orange fruit",
        "host_tokens": "tabletop bowl",
        "support_mask": "outputs/phase2_dece_flux_custom_masks_20260616/white_bowl_orange_table_v2.png",
        "support_control_mode": "fixed",
        "params": {
            "edit_hedit": 1.00,
            "edit_region": 0.14,
            "edit_target": 0.20,
            "local_target": 0.90,
            "region_transport": 0.10,
            "rec": 0.22,
            "struct": 0.45,
            "traj": 0.28,
            "preserve_gain": 3.0,
        },
    },
    "brown_bowl_lemon_phase2": {
        "op": "add_object",
        "relation": "inside_container",
        "kind": "add",
        "local_prompt": "One single yellow lemon inside a brown bowl.",
        "new_tokens": "lemon yellow",
        "host_tokens": "bowl inside",
    },
    "tshirt_star": {
        "op": "add_decal",
        "relation": "on_surface",
        "kind": "decal",
        "overlay": "star",
        "local_prompt": "One single flat red star printed at the center front of a white T-shirt.",
        "local_target_prompt": "The same person wearing the same white T-shirt and blue jeans, with one single clearly visible medium-sized bright red star printed at the center chest, while preserving the fabric folds, shadows, jeans, pose, and background.",
        "new_tokens": "star red",
        "host_tokens": "shirt t-shirt front",
        "params": {
            "edit_hedit": 0.64,
            "edit_region": 0.18,
            "edit_target": 0.06,
            "local_target": 0.44,
            "region_transport": 0.10,
            "rec": 0.42,
            "struct": 0.45,
            "traj": 0.30,
            "preserve_gain": 3.8,
        },
    },
    "mug_heart": {
        "op": "add_decal",
        "relation": "on_surface",
        "kind": "decal",
        "overlay": "heart",
        "local_prompt": "Exactly one small flat red heart decal centered on the front of a white ceramic mug, with no other hearts.",
        "target_prompt": "The same white ceramic mug in the same studio scene, with exactly one small flat red heart decal centered on the front of the mug and no other markings.",
        "local_target_prompt": "The same white ceramic mug with exactly one small flat red heart centered on the front, no other markings.",
        "new_tokens": "heart red",
        "host_tokens": "mug ceramic front",
        "support_mask": "outputs/phase2_dece_flux_custom_masks_20260616/mug_heart_center.png",
        "support_control_mode": "fixed",
        "params": {
            "edit_hedit": 0.98,
            "edit_region": 0.34,
            "edit_target": 0.20,
            "local_target": 0.95,
            "rec": 0.24,
            "struct": 0.40,
            "traj": 0.16,
            "preserve_gain": 3.2,
            "final_mask_alpha_gamma": 0.55,
        },
    },
    "tote_leaf": {
        "op": "add_decal",
        "relation": "on_surface",
        "kind": "decal",
        "overlay": "leaf",
        "local_prompt": "One single flat green leaf decal printed on the front of a tote bag.",
        "new_tokens": "leaf green",
        "host_tokens": "tote bag front",
        "params": {
            "edit_hedit": 1.00,
            "edit_region": 0.16,
            "edit_target": 0.18,
            "local_target": 0.70,
            "region_transport": 0.08,
            "rec": 0.28,
            "struct": 0.45,
            "traj": 0.30,
            "preserve_gain": 3.8,
        },
    },
    "red_office_chair_to_blue_office_chair": {"op": "recolor", "relation": "inside", "kind": "recolor", "color": "blue", "source_color": "red"},
    "green_mug_orange_phase2": {"op": "recolor", "relation": "inside", "kind": "recolor", "color": "orange", "source_color": "green"},
    "yellow_vase_blue_phase2": {"op": "recolor", "relation": "inside", "kind": "recolor", "color": "blue", "source_color": "yellow", "recolor_full_mask": True, "params": {"final_mask_dilate": 1}},
    "pillow_same_color_cable_knit": {
        "op": "add_decal",
        "relation": "on_surface",
        "kind": "material",
        "source_color": "light_neutral",
        "params": {"core_attention_percentile": 12.0, "edit_hedit": 0.85, "local_target": 0.55, "edit_target": 0.12, "edit_region": 0.12, "region_transport": 0.10, "rec": 0.38, "struct": 0.45, "preserve_gain": 2.0},
        "local_prompt": "Same-color white chunky cable-knit fabric covering the entire pillow surface, with thick braided knitted columns and fine all-over knit stitches, photorealistic knitted wool, not checkerboard, no text.",
        "local_target_prompt": "A photo of the same white pillow on the same brown sofa, with the entire pillow surface changed into same-color white chunky cable-knit fabric with thick braided knitted columns, while preserving the pillow shape, outline, lighting, sofa, background, and the rest of the scene.",
        "new_tokens": "cable knit knitted braided",
        "host_tokens": "pillow fabric",
    },
    "pillow_same_color_cable_knit_grey": {
        "op": "add_decal",
        "relation": "on_surface",
        "kind": "material",
        "source_color": "grey",
        "local_prompt": "Texture-only grey chunky cable-knit braided stitch fabric covering the pillow surface.",
        "local_target_prompt": "The same pillow in the same scene, with the pillow surface changed into texture-only grey chunky cable-knit braided stitch fabric, preserving the pillow shape, outline, lighting, sofa, background, and the rest of the scene.",
        "new_tokens": "cable knit knitted braided",
        "host_tokens": "pillow fabric",
    },
    "pillow_same_color_cable_knit_armchair": {
        "op": "add_decal",
        "relation": "on_surface",
        "kind": "material",
        "source_color": "light_neutral",
        "params": {"core_attention_percentile": 12.0, "edit_hedit": 0.85, "local_target": 0.55, "edit_target": 0.12, "edit_region": 0.12, "region_transport": 0.10, "rec": 0.38, "struct": 0.45, "preserve_gain": 2.0},
        "local_prompt": "Same-color white chunky cable-knit fabric covering the entire armchair pillow surface, with thick braided knitted columns and fine all-over knit stitches, photorealistic knitted wool, not checkerboard, no text.",
        "local_target_prompt": "A photo of the same armchair pillow in the same scene, with the entire pillow surface changed into same-color white chunky cable-knit fabric with thick braided knitted columns, while preserving the pillow shape, outline, lighting, armchair, background, and the rest of the scene.",
        "new_tokens": "cable knit knitted braided",
        "host_tokens": "pillow fabric",
    },
}

KIND_PARAMS = {
    "add": {
        "edit_hedit": 0.95,
        "edit_anchor": 0.18,
        "edit_region": 0.32,
        "edit_target": 0.18,
        "edit_source": 0.02,
        "local_target": 0.95,
        "region_transport": 0.35,
        "core_attention_percentile": 82.0,
        "outside_lock": 0.03,
        "rec": 0.18,
        "struct": 0.40,
        "traj": 0.12,
        "preserve_budget": 0.20,
        "preserve_gain": 2.2,
        "final_mask_alpha_gamma": 0.75,
    },
    "decal": {
        "edit_hedit": 0.98,
        "edit_anchor": 0.18,
        "edit_region": 0.34,
        "edit_target": 0.18,
        "edit_source": 0.015,
        "local_target": 0.90,
        "region_transport": 0.24,
        "core_attention_percentile": 86.0,
        "outside_lock": 0.0,
        "rec": 0.28,
        "struct": 0.40,
        "traj": 0.18,
        "preserve_budget": 0.14,
        "preserve_gain": 3.2,
        "final_mask_alpha_gamma": 0.65,
    },
    "material": {
        "edit_hedit": 1.30,
        "edit_anchor": 0.14,
        "edit_region": 0.30,
        "edit_target": 0.26,
        "edit_source": 0.01,
        "local_target": 1.05,
        "region_transport": 0.16,
        "core_attention_percentile": 20.0,
        "outside_lock": 0.15,
        "rec": 0.15,
        "struct": 0.40,
        "traj": 0.10,
        "preserve_budget": 0.12,
        "preserve_gain": 1.6,
        "final_knit_texture": 6.0,
        "final_knit_source_blend": 0.58,
        "final_mask_dilate": 1,
        "final_mask_alpha_gamma": 0.70,
    },
    "recolor": {
        "edit_hedit": 0.36,
        "edit_anchor": 0.06,
        "edit_region": 0.14,
        "edit_target": 0.04,
        "edit_source": 0.0,
        "local_target": 0.18,
        "region_transport": 0.12,
        "outside_lock": 0.10,
        "rec": 0.50,
        "struct": 0.45,
        "traj": 0.40,
        "preserve_budget": 0.10,
        "preserve_gain": 3.0,
        "recolor_projection": 0.75,
        "final_recolor_blend": 0.78,
        "final_mask_dilate": 0,
        "final_mask_alpha_gamma": 0.60,
    },
}


def read_manifest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = str(row.get("seed", "")).removeprefix("seed_")
            if seed:
                rows.setdefault((row["task"], seed), row)
    return rows


def support_mask(project: Path, task: str, seed: str) -> Path:
    return project / SD3_SUPPORT_ROOT / task / "support_v3_controller_rmsgap" / f"seed_{seed}" / "masks" / "semantic_support.png"


def _current_hostnames() -> set[str]:
    names = {socket.gethostname()}
    try:
        fqdn = subprocess.check_output(["hostname", "-f"], text=True, timeout=5).strip()
    except Exception:
        fqdn = ""
    if fqdn:
        names.add(fqdn)
    for key in ("HOSTNAME", "SLURMD_NODENAME", "SLURM_JOB_NODELIST", "SLURM_NODELIST"):
        value = os.environ.get(key)
        if not value:
            continue
        names.add(value)
        if "a100-01" in value:
            names.add("a100-01.gpu01.cis.k.hosei.ac.jp")
        if "h100-01" in value:
            names.add("h100-01.gpu01.cis.k.hosei.ac.jp")
    return names


GPU_COMPUTE_HOSTS = {
    "a100-01.gpu01.cis.k.hosei.ac.jp",
    "h100-01.gpu01.cis.k.hosei.ac.jp",
}


def command_for(
    *,
    project: Path,
    row: dict[str, str],
    task: str,
    seed: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cfg = TASK_CFG[task]
    params = {**KIND_PARAMS[cfg["kind"]], **cfg.get("params", {})}
    mask = project / cfg["support_mask"] if cfg.get("support_mask") else support_mask(project, task, seed)
    cmd = [
        str(project / ".venv" / "bin" / "python"),
        str(project / "scripts" / "run_dece_image.py"),
        "--backend",
        "flux",
        "--project",
        str(project),
        "--python",
        str(project / ".venv" / "bin" / "python"),
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
        "--target-prompt",
        cfg.get("target_prompt", row["target_prompt"]),
        "--method",
        "dece_rf_flux",
        "--seed",
        seed,
        "--num-inference-steps",
        str(args.steps),
        "--n-max",
        str(args.n_max),
        "--max-image-size",
        str(args.max_image_size),
        "--max-sequence-length",
        str(args.max_sequence_length),
        "--src-guidance-scale",
        str(args.src_guidance),
        "--base-guidance-scale",
        str(args.src_guidance),
        "--tar-guidance-scale",
        str(args.tar_guidance),
        "--support-mask",
        str(mask),
        "--support-control-mode",
        cfg.get("support_control_mode", "fixed"),
        "--support-external-mask-role",
        cfg.get("support_external_mask_role", "attention"),
        "--edit-operation",
        cfg["op"],
        "--support-relation",
        cfg["relation"],
        "--mask-layering-mode",
        "object_contact",
        "--edit-hedit-guidance-scale",
        str(params["edit_hedit"]),
        "--edit-guidance-scale",
        str(params["edit_anchor"]),
        "--edit-region-guidance-scale",
        str(params["edit_region"]),
        "--edit-target-guidance-scale",
        str(params["edit_target"]),
        "--edit-source-guidance-scale",
        str(params["edit_source"]),
        "--edit-local-target-prompt",
        cfg.get("local_target_prompt", row["target_prompt"]),
        "--edit-local-target-guidance-scale",
        str(params["local_target"]),
        "--edit-local-target-cfg-scale",
        str(args.tar_guidance),
        "--rec-guidance-scale",
        str(params["rec"]),
        "--struct-guidance-scale",
        str(params["struct"]),
        "--trajectory-preserve-scale",
        str(params["traj"]),
        "--beta-max",
        "1.0",
        "--rec-stop-timestep",
        "0.08",
        "--linear-path-t-min",
        "0.05",
        "--adaptive-clean-control",
        "--adaptive-edit-target-rms",
        str(args.adaptive_edit_target_rms),
        "--adaptive-rmsgap-mode",
        "legacy",
        "--adaptive-preserve-drift-budget",
        str(params["preserve_budget"]),
        "--adaptive-edit-gain",
        "2.0",
        "--adaptive-preserve-gain",
        str(params["preserve_gain"]),
        "--adaptive-edit-weight-min",
        "0.85",
        "--adaptive-edit-weight-max",
        "1.55",
        "--adaptive-preserve-weight-min",
        "1.0",
        "--adaptive-preserve-weight-max",
        "1.65",
        "--adaptive-projection-scale",
        "0.65",
        "--adaptive-preserve-clean-correction-scale",
        "0.5",
        "--region-target-transport-scale",
        str(params["region_transport"]),
        "--region-target-outside-lock-scale",
        str(params["outside_lock"]),
        "--final-postprocess-mode",
        "debug_visual" if args.debug_final_visual_postprocess else "mask_blend",
        "--final-mask-blend-scale",
        str(params.get("final_mask_blend_scale", args.final_mask_blend_scale)),
        "--final-mask-alpha-gamma",
        str(params.get("final_mask_alpha_gamma", args.final_mask_alpha_gamma)),
        "--mask-output-dir",
        str(out_dir / "masks"),
        "--output",
        str(out_dir / "result.png"),
        "--metadata-output",
        str(out_dir / "metadata.json"),
        "--stats-output",
        str(out_dir / "stats.json"),
    ]
    if cfg.get("support_candidate"):
        cmd.extend(["--support-candidate", str(cfg["support_candidate"])])
    if cfg.get("support_top_percentile") is not None:
        cmd.extend(["--support-top-percentile", str(cfg["support_top_percentile"])])
    if cfg.get("support_dilate_radius") is not None:
        cmd.extend(["--support-dilate-radius", str(cfg["support_dilate_radius"])])
    if cfg.get("support_max_area_ratio") is not None:
        cmd.extend(["--support-max-area-ratio", str(cfg["support_max_area_ratio"])])
    if params.get("core_attention_percentile") is not None:
        cmd.extend(
            [
                "--fixed-core-from-attention",
                "--fixed-core-attention-percentile",
                str(params["core_attention_percentile"]),
            ]
        )
    if params["edit_target"] > 0.0 or params["edit_source"] > 0.0:
        cmd.append("--use-flux-attention-support")
        if cfg.get("new_tokens"):
            cmd.extend(["--new-tokens", str(cfg["new_tokens"])])
        if cfg.get("host_tokens"):
            cmd.extend(["--host-tokens", str(cfg["host_tokens"])])
    if cfg["kind"] == "recolor":
        cmd.extend(
            [
                "--recolor-target",
                cfg["color"],
                "--recolor-clean-projection-scale",
                str(params["recolor_projection"]),
                "--final-recolor-blend-scale",
                str(params["final_recolor_blend"]),
                "--final-mask-dilate",
                str(params["final_mask_dilate"]),
            ]
        )
        if cfg.get("source_color"):
            cmd.extend(["--final-source-color-mask", cfg["source_color"]])
    if cfg["kind"] == "material":
        cmd.extend(
            [
                "--final-knit-texture-scale",
                str(params["final_knit_texture"]),
                "--final-knit-source-blend",
                str(params["final_knit_source_blend"]),
                "--final-mask-dilate",
                str(params["final_mask_dilate"]),
            ]
        )
        if cfg.get("source_color"):
            cmd.extend(["--final-source-color-mask", cfg["source_color"]])
    if cfg.get("overlay"):
        cmd.extend(["--final-object-overlay", str(cfg["overlay"])])
    if getattr(args, "true_cfg", False):
        cmd.extend(["--extra-arg=--true-cfg", f"--extra-arg=--distilled-guidance {args.distilled_guidance}"])
    if cfg.get("recolor_full_mask"):
        cmd.append("--extra-arg=--recolor-full-mask")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--tasks", default=" ".join(TASKS))
    parser.add_argument("--seeds", default="10")
    parser.add_argument("--output-root", default="outputs/phase2_dece_flux_paper_seed10_v1")
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--src-guidance", type=float, default=1.0)
    parser.add_argument("--tar-guidance", type=float, default=10.5)
    parser.add_argument("--true-cfg", action="store_true")
    parser.add_argument("--distilled-guidance", type=float, default=1.0)
    parser.add_argument("--adaptive-edit-target-rms", type=float, default=0.42)
    parser.add_argument("--final-mask-blend-scale", type=float, default=1.0)
    parser.add_argument("--final-mask-alpha-gamma", type=float, default=1.0)
    parser.add_argument("--debug-final-visual-postprocess", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    hostnames = _current_hostnames()
    if not args.dry_run and hostnames.isdisjoint(GPU_COMPUTE_HOSTS):
        raise SystemExit(f"refusing heavy FLUX run on {sorted(hostnames)}")

    rows: dict[tuple[str, str], dict[str, str]] = {}
    rows.update(read_manifest(project / T1_T4_MANIFEST))
    rows.update(read_manifest(project / T5_MANIFEST))

    selected = [item for item in args.tasks.split() if item]
    seeds = [str(item).removeprefix("seed_") for item in args.seeds.split() if item]
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
    for seed in seeds:
        for task in selected:
            row = rows.get((task, seed))
            if row is None:
                raise SystemExit(f"missing seed{seed} manifest row for {task}")
            mask = support_mask(project, task, seed)
            if not mask.is_file():
                raise SystemExit(f"missing SD3 support mask for {task} seed {seed}: {mask}")
            out_dir = output_root / task / "dece_rf_flux" / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = command_for(project=project, row=row, task=task, seed=seed, out_dir=out_dir, args=args)
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
                    summary.append({"task": task, "seed": seed, "status": status, "out_dir": str(out_dir)})
                    break
            summary.append({"task": task, "seed": seed, "status": status, "out_dir": str(out_dir)})
        if summary and str(summary[-1]["status"]).startswith("failed:"):
            break

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    for item in summary:
        print(item["task"], f"seed_{item['seed']}", item["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
