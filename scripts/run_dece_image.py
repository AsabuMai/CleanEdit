from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


COMMON_VALUE_ARGS = (
    "image",
    "source_prompt",
    "output",
    "seed",
    "num_inference_steps",
    "n_max",
    "max_image_size",
    "src_guidance_scale",
    "tar_guidance_scale",
    "base_guidance_scale",
    "inversion_guidance_scale",
    "edit_src_cfg_scale",
    "rec_guidance_scale",
    "struct_guidance_scale",
    "edit_hedit_guidance_scale",
    "edit_guidance_scale",
    "edit_region_guidance_scale",
    "edit_target_guidance_scale",
    "edit_source_guidance_scale",
    "edit_core_scale",
    "edit_subject_scale",
    "edit_local_target_prompt",
    "edit_local_target_guidance_scale",
    "edit_local_target_cfg_scale",
    "alpha_max",
    "alpha_schedule",
    "beta_max",
    "beta_schedule",
    "adaptive_edit_target_rms",
    "adaptive_rmsgap_dead_zone",
    "adaptive_rmsgap_preserve_gate_budget",
    "adaptive_preserve_drift_budget",
    "adaptive_edit_gain",
    "adaptive_preserve_gain",
    "adaptive_edit_weight_min",
    "adaptive_edit_weight_max",
    "adaptive_preserve_weight_min",
    "adaptive_preserve_weight_max",
    "adaptive_projection_scale",
    "adaptive_preserve_clean_correction_scale",
    "trajectory_preserve_scale",
    "trajectory_subject_preserve_scale",
    "region_target_transport_scale",
    "region_target_outside_lock_scale",
    "support_mask",
    "support_mode",
    "object_mask_provider",
    "support_score",
    "support_candidate",
    "edit_operation",
    "support_relation",
    "grounding_method",
    "new_tokens",
    "host_tokens",
    "removed_tokens",
    "support_top_percentile",
    "support_min_area_ratio",
    "support_max_area_ratio",
    "support_keep_components",
    "support_dilate_radius",
    "support_blur_kernel",
    "mask_layering_mode",
    "mask_object_threshold",
    "mask_contact_dilate_kernel",
    "mask_contact_scale",
    "mask_contact_edge_threshold",
    "mask_contact_edge_protect_scale",
    "mask_output_dir",
    "mask_blend_mode",
    "stats_output",
    "metadata_output",
)

COMMON_FLAG_ARGS = (
    "adaptive_clean_control",
    "save_support_debug",
    "support_debug_only",
    "mask_blend",
)

FLUX_VALUE_ARGS = (
    "model_id",
    "cache_dir",
    "method",
    "torch_dtype",
    "guidance_scale",
    "support_control_mode",
    "support_external_mask_role",
    "recolor_target",
    "recolor_clean_projection_scale",
    "linear_path_t_min",
    "rec_stop_timestep",
    "adaptive_rmsgap_mode",
    "final_postprocess_mode",
    "final_mask_blend_scale",
    "final_mask_alpha_gamma",
    "final_recolor_blend_scale",
    "final_knit_texture_scale",
    "final_knit_source_blend",
    "final_mask_dilate",
    "final_source_color_mask",
    "final_object_overlay",
    "fixed_core_attention_percentile",
    "max_sequence_length",
)

FLUX_FLAG_ARGS = (
    "local_files_only",
    "model_offload",
    "use_flux_attention_support",
    "fixed_core_from_attention",
)

SD3_VALUE_ARGS = (
    "eta",
    "support_temporal_aggregation",
    "attention_mask_mode",
    "edit_color_target",
    "edit_color_clean_projection_scale",
)

SD3_FLAG_ARGS = (
    "low_vram",
)

ALIASES = {
    "target_prompt": "prompt",
    "support_mask": "support_mask",
    "mask_layering_mode": "mask_layering_mode",
}


def _add_value_arg(parser: argparse.ArgumentParser, name: str, *, arg_type=str, default=None) -> None:
    parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=arg_type, default=default)


def _add_flag_arg(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(f"--{name.replace('_', '-')}", dest=name, action="store_true", default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one CleanEdit/HRec edit through the SD3 or FLUX backend using a shared interface."
    )
    parser.add_argument("--backend", choices=("sd3", "flux"), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--command-output", type=Path, default=None)
    parser.add_argument("--record-output", type=Path, default=None)
    parser.add_argument("--extra-arg", action="append", default=[], help="Raw backend arg, repeatable.")

    _add_value_arg(parser, "image", default=None)
    _add_value_arg(parser, "source_prompt", default=None)
    parser.add_argument("--target-prompt", "--prompt", dest="target_prompt", default=None)
    _add_value_arg(parser, "output", default=None)
    _add_value_arg(parser, "seed", arg_type=int, default=None)
    _add_value_arg(parser, "num_inference_steps", arg_type=int, default=None)
    _add_value_arg(parser, "n_max", arg_type=int, default=None)
    _add_value_arg(parser, "max_image_size", arg_type=int, default=None)

    for name in (
        "src_guidance_scale",
        "tar_guidance_scale",
        "base_guidance_scale",
        "inversion_guidance_scale",
        "edit_src_cfg_scale",
        "rec_guidance_scale",
        "struct_guidance_scale",
        "edit_hedit_guidance_scale",
        "edit_guidance_scale",
        "edit_region_guidance_scale",
        "edit_target_guidance_scale",
        "edit_source_guidance_scale",
        "edit_core_scale",
        "edit_subject_scale",
        "edit_local_target_guidance_scale",
        "edit_local_target_cfg_scale",
        "alpha_max",
        "beta_max",
        "adaptive_edit_target_rms",
        "adaptive_rmsgap_dead_zone",
        "adaptive_rmsgap_preserve_gate_budget",
        "adaptive_preserve_drift_budget",
        "adaptive_edit_gain",
        "adaptive_preserve_gain",
        "adaptive_edit_weight_min",
        "adaptive_edit_weight_max",
        "adaptive_preserve_weight_min",
        "adaptive_preserve_weight_max",
        "adaptive_projection_scale",
        "adaptive_preserve_clean_correction_scale",
        "trajectory_preserve_scale",
        "trajectory_subject_preserve_scale",
        "region_target_transport_scale",
        "region_target_outside_lock_scale",
        "support_top_percentile",
        "support_min_area_ratio",
        "support_max_area_ratio",
        "mask_object_threshold",
        "mask_contact_scale",
        "mask_contact_edge_threshold",
        "mask_contact_edge_protect_scale",
        "guidance_scale",
        "recolor_clean_projection_scale",
        "linear_path_t_min",
        "rec_stop_timestep",
        "final_mask_blend_scale",
        "final_mask_alpha_gamma",
        "final_recolor_blend_scale",
        "final_knit_texture_scale",
        "final_knit_source_blend",
        "fixed_core_attention_percentile",
        "eta",
        "edit_color_clean_projection_scale",
    ):
        _add_value_arg(parser, name, arg_type=float, default=None)

    for name in (
        "support_keep_components",
        "support_dilate_radius",
        "support_blur_kernel",
        "mask_contact_dilate_kernel",
        "final_mask_dilate",
        "max_sequence_length",
    ):
        _add_value_arg(parser, name, arg_type=int, default=None)

    for name in (
        "edit_local_target_prompt",
        "alpha_schedule",
        "beta_schedule",
        "support_mask",
        "support_mode",
        "object_mask_provider",
        "support_score",
        "support_candidate",
        "edit_operation",
        "support_relation",
        "grounding_method",
        "new_tokens",
        "host_tokens",
        "removed_tokens",
        "mask_layering_mode",
        "mask_output_dir",
        "mask_blend_mode",
        "stats_output",
        "metadata_output",
        "model_id",
        "cache_dir",
        "method",
        "torch_dtype",
        "support_control_mode",
        "support_external_mask_role",
        "recolor_target",
        "adaptive_rmsgap_mode",
        "final_postprocess_mode",
        "final_source_color_mask",
        "final_object_overlay",
        "support_temporal_aggregation",
        "attention_mask_mode",
        "edit_color_target",
    ):
        _add_value_arg(parser, name, default=None)

    for name in COMMON_FLAG_ARGS + FLUX_FLAG_ARGS + SD3_FLAG_ARGS:
        _add_flag_arg(parser, name)
    return parser


def _append_values(cmd: list[str], args: argparse.Namespace, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(args, name)
        if value is None:
            continue
        cli_name = ALIASES.get(name, name).replace("_", "-")
        cmd.extend([f"--{cli_name}", str(value)])


def _append_flags(cmd: list[str], args: argparse.Namespace, names: tuple[str, ...]) -> None:
    for name in names:
        if getattr(args, name):
            cmd.append(f"--{name.replace('_', '-')}")


def build_backend_command(args: argparse.Namespace) -> list[str]:
    missing = [
        name
        for name in ("image", "source_prompt", "target_prompt", "output")
        if not getattr(args, name)
    ]
    if missing:
        raise SystemExit(f"missing required shared args: {', '.join('--' + item.replace('_', '-') for item in missing)}")

    entrypoint = "run_edit_sd3.py" if args.backend == "sd3" else "run_edit_flux.py"
    cmd = [args.python, str(args.project / entrypoint)]
    _append_values(cmd, args, COMMON_VALUE_ARGS)
    cmd.extend(["--prompt", str(args.target_prompt)])
    _append_flags(cmd, args, COMMON_FLAG_ARGS)

    if args.backend == "sd3":
        _append_values(cmd, args, SD3_VALUE_ARGS)
        _append_flags(cmd, args, SD3_FLAG_ARGS)
    else:
        if args.method is None:
            cmd.extend(["--method", "dece_rf_flux"])
        _append_values(cmd, args, FLUX_VALUE_ARGS)
        _append_flags(cmd, args, FLUX_FLAG_ARGS)

    for item in args.extra_arg:
        cmd.extend(shlex.split(item))
    return cmd


def main() -> int:
    args = build_parser().parse_args()
    args.project = args.project.resolve()
    cmd = build_backend_command(args)
    command_text = subprocess.list2cmdline(cmd)
    if args.command_output:
        args.command_output.parent.mkdir(parents=True, exist_ok=True)
        args.command_output.write_text(command_text + "\n", encoding="utf-8")
    record = {
        "backend": args.backend,
        "entrypoint": "run_edit_sd3.py" if args.backend == "sd3" else "run_edit_flux.py",
        "image": args.image,
        "source_prompt": args.source_prompt,
        "target_prompt": args.target_prompt,
        "output": args.output,
        "command": command_text,
    }
    if args.record_output:
        args.record_output.parent.mkdir(parents=True, exist_ok=True)
        args.record_output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(command_text)
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=args.project).returncode


if __name__ == "__main__":
    raise SystemExit(main())
