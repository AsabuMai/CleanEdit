from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/cluster/users/grad/2025/25t8103/project")
REPAIR_ROOT = ROOT / "data/flowedit_compatible_135/repair_allpass_v1"
SD3_OUT = ROOT / "outputs/fe135_allpass_sd3_v1"
FLUX_OUT = ROOT / "outputs/fe135_allpass_flux_v1"

SD3_REUSE_OLD_KEYS = {
    "fe_046_cat_crown_1_black_top_hat",
    "fe_094_dog_6_red_top_hat",
    "fe_142_iguana_2_blue_lizard_top_hat",
    "fe_180_milk_4_whipped_cream",
}

FLUX_REUSE_OLD_KEYS = {
    "fe_201_pizza_1_pineapple_ham",
    "fe_207_pizza_tomato_olive_1_pepperoni",
    "fe_208_pizza_tomato_olive_2_mushrooms",
}

ALLOWED_RELATIONS = {
    "auto",
    "none",
    "above_host",
    "below_host",
    "on_face",
    "on_profile_face",
    "on_profile_face_left",
    "on_profile_face_right",
    "on_surface",
    "remove_source_object",
    "inside_host",
    "inside_object",
    "inside",
    "inside_container",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def add(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def support_score_for(category: str) -> str:
    if category == "C1_replacement_old_object_residue":
        return "removed_src_x_clean"
    if category in {"C2_small_or_compound_accessory", "C3_insertion_mask_geometry"}:
        return "host_top_contact"
    if category == "C4_insertion_strength_balance":
        return "new_x_surface_local_response"
    if category == "C7_low_contrast_or_small_recolor":
        return "seg_x_response"
    return "attention_x_clean"


def sd3_operation_relation(category: str, operation: str, relation: str) -> tuple[str, str]:
    if category == "C1_replacement_old_object_residue":
        return "add_object", "inside_host"
    if category == "C3_insertion_mask_geometry":
        return "add_object", "inside_container"
    return operation, relation


def normalize_relation(relation: str | None, operation: str | None, key: str = "") -> str:
    rel = (relation or "auto").strip()
    if rel in ALLOWED_RELATIONS:
        return rel
    low = rel.lower().replace("-", "_")
    if low in {"top", "top_center", "head_top", "on_top", "above", "upper"}:
        return "above_host"
    if low in {"face", "head", "front_face"}:
        return "on_face"
    if low in {"surface", "table", "ground"}:
        return "on_surface"
    if low in {"inside", "inner", "whole_object"}:
        return "inside"
    op = (operation or "").lower()
    if op == "add_object":
        return "above_host" if any(token in key for token in ("hat", "crown")) else "on_surface"
    if op in {"replace", "recolor"}:
        return "inside_host" if op == "replace" else "inside"
    return "auto"


def sd3_params(category: str, item: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "num_inference_steps": 28,
        "n_max": 24,
        "src_guidance_scale": 1.0,
        "tar_guidance_scale": 8.0,
        "inversion_guidance_scale": 1.0,
        "base_guidance_scale": 1.0,
        "edit_src_cfg_scale": 1.0,
        "rec_guidance_scale": 0.26,
        "struct_guidance_scale": 0.45,
        "edit_hedit_guidance_scale": 0.95,
        "edit_text_guidance_scale": 0.1,
        "edit_core_scale": 1.0,
        "edit_subject_scale": 0.25,
        "alpha_max": 0.26,
        "beta_max": 1.0,
        "trajectory_preserve_scale": 0.25,
        "object_mask_provider": "operation_support_v3",
        "support_mode": "operation_v3",
        "support_score": support_score_for(category),
        "support_candidate": support_score_for(category),
        "mask_layering_mode": "object_contact",
        "save_support_debug": True,
        "adaptive_clean_control": True,
        "adaptive_edit_target_rms": 0.42,
        "adaptive_edit_gain": 2.0,
        "adaptive_preserve_gain": 2.5,
        "adaptive_projection_scale": 0.65,
    }
    if category == "C1_replacement_old_object_residue":
        params.update(
            {
                "tar_guidance_scale": 8.0,
                "edit_text_guidance_scale": 0.12,
                "edit_core_scale": 1.0,
                "edit_subject_scale": 0.15,
                "trajectory_preserve_scale": 0.25,
                "support_score": "attention_x_clean",
                "support_candidate": "attention_x_clean",
                "mask_layering_mode": "none",
            }
        )
    elif category == "C2_small_or_compound_accessory":
        params.update(
            {
                "tar_guidance_scale": 8.8,
                "edit_text_guidance_scale": 0.28,
                "edit_core_scale": 1.20,
                "edit_subject_scale": 0.18,
                "support_min_area_ratio": 0.006,
                "support_max_area_ratio": 0.18,
            }
        )
    elif category == "C3_insertion_mask_geometry":
        params.update(
            {
                "tar_guidance_scale": 8.0,
                "edit_core_scale": 1.0,
                "edit_subject_scale": 0.25,
                "edit_initial_noise_scale": 0.0,
                "edit_initial_noise_region": "core",
                "region_target_transport_scale": 0.0,
                "region_target_outside_lock_scale": 0.0,
                "support_score": "attention_x_clean",
                "support_candidate": "attention_x_clean",
                "mask_layering_mode": "none",
            }
        )
    elif category == "C4_insertion_strength_balance":
        params.update(
            {
                "tar_guidance_scale": 8.4,
                "edit_core_scale": 1.12,
                "edit_subject_scale": 0.22,
                "region_target_transport_scale": 0.18,
                "region_target_outside_lock_scale": 0.05,
            }
        )
    return params


def flux_params(category: str, item: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "method": "dece_rf_flux",
        "model_id": "black-forest-labs/FLUX.1-dev",
        "cache_dir": str(ROOT / ".cache/huggingface/hub"),
        "local_files_only": True,
        "num_inference_steps": 12,
        "n_max": 10,
        "guidance_scale": 3.5,
        "src_guidance_scale": 1.0,
        "tar_guidance_scale": 5.0,
        "base_guidance_scale": 1.0,
        "inversion_guidance_scale": 1.0,
        "edit_src_cfg_scale": 1.0,
        "rec_guidance_scale": 0.18,
        "struct_guidance_scale": 0.4,
        "edit_hedit_guidance_scale": 0.95,
        "edit_guidance_scale": 0.20,
        "edit_region_guidance_scale": 0.32,
        "edit_target_guidance_scale": 0.20,
        "edit_source_guidance_scale": 0.02,
        "edit_core_scale": 1.35,
        "edit_subject_scale": 0.35,
        "edit_local_target_guidance_scale": 0.95,
        "edit_local_target_cfg_scale": 5.0,
        "adaptive_clean_control": True,
        "adaptive_edit_target_rms": 0.42,
        "adaptive_edit_gain": 2.0,
        "adaptive_preserve_gain": 2.5,
        "adaptive_projection_scale": 0.65,
        "trajectory_preserve_scale": 0.12,
        "region_target_transport_scale": 0.35,
        "region_target_outside_lock_scale": 0.03,
        "final_postprocess_mode": "mask_blend",
        "final_mask_blend_scale": 1.0,
        "support_control_mode": "fixed",
        "support_mode": "fixed",
        "support_external_mask_role": "attention",
        "support_candidate": "operation_default",
    }
    if category == "C1_replacement_old_object_residue":
        params.update({"edit_guidance_scale": 0.26, "edit_core_scale": 1.55, "edit_subject_scale": 0.20, "final_mask_dilate": 3})
    elif category == "C2_small_or_compound_accessory":
        params.update({"edit_guidance_scale": 0.28, "edit_core_scale": 1.60, "edit_subject_scale": 0.24, "final_mask_dilate": 2})
    elif category == "C3_insertion_mask_geometry":
        params.update({"edit_guidance_scale": 0.30, "edit_core_scale": 1.65, "edit_subject_scale": 0.22, "final_mask_alpha_gamma": 0.82})
    elif category == "C4_insertion_strength_balance":
        params.update({"edit_guidance_scale": 0.24, "edit_core_scale": 1.48, "edit_subject_scale": 0.28, "final_mask_alpha_gamma": 0.90})
    return params


def output_dir(backend: str, entry: dict[str, Any], item: dict[str, Any]) -> Path:
    if backend == "sd3":
        return SD3_OUT / entry["key"] / item["sd3_recipe"] / "seed_10"
    return FLUX_OUT / entry["key"] / item["flux_recipe"] / "seed_10"


def build_cmd(backend: str, entry: dict[str, Any], item: dict[str, Any]) -> list[str]:
    out_dir = output_dir(backend, entry, item)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    image = resolve(entry["image"])
    mask = resolve(item["mask_path"])
    prompt = item["prompt_override"]
    negative = item.get("negative_prompt") or ""
    operation = item.get("edit_operation") or entry.get("edit_operation") or "auto"
    relation = normalize_relation(item.get("support_relation"), operation, entry["key"])
    if backend == "sd3":
        cmd = [sys.executable, str(ROOT / "run_edit_sd3.py")]
        params = sd3_params(item["category"], item)
        operation, relation = sd3_operation_relation(item["category"], operation, relation)
        add(cmd, "--image", image)
        add(cmd, "--source-prompt", entry["source_prompt"])
        add(cmd, "--prompt", prompt)
        add(cmd, "--negative-prompt", negative)
        add(cmd, "--output", out_dir / "result.png")
        add(cmd, "--stats-output", out_dir / "stats.json")
        add(cmd, "--metadata-output", out_dir / "metadata.json")
        add(cmd, "--mask-output-dir", out_dir / "masks")
        add(cmd, "--support-mask", mask)
        add(cmd, "--semantic-base-mask", mask)
        add(cmd, "--edit-operation", operation)
        add(cmd, "--support-relation", relation)
        add(cmd, "--seed", 10)
        for key, value in params.items():
            add(cmd, "--" + key.replace("_", "-"), value)
        return cmd

    cmd = [sys.executable, "-m", "flux.run_edit_flux"]
    params = flux_params(item["category"], item)
    add(cmd, "--image", image)
    add(cmd, "--source-prompt", entry["source_prompt"])
    add(cmd, "--prompt", prompt)
    add(cmd, "--negative-prompt", negative)
    add(cmd, "--output", out_dir / "result.png")
    add(cmd, "--stats-output", out_dir / "stats.json")
    add(cmd, "--metadata-output", out_dir / "metadata.json")
    add(cmd, "--support-mask", mask)
    add(cmd, "--semantic-base-mask", mask)
    add(cmd, "--edit-operation", operation)
    add(cmd, "--support-relation", relation)
    add(cmd, "--edit-local-target-prompt", prompt)
    add(cmd, "--seed", 10)
    for key, value in params.items():
        add(cmd, "--" + key.replace("_", "-"), value)
    return cmd


def apply_postprocess_if_needed(backend: str, entry: dict[str, Any], item: dict[str, Any]) -> None:
    if item.get("postprocess_branch") in (None, "", "none"):
        return
    sys.path.insert(0, str(ROOT / "scripts/allpass_v1"))
    from build_allpass_v1 import apply_cpu_postprocess

    apply_cpu_postprocess(output_dir(backend, entry, item), entry, item, backend)


def should_run(entry: dict[str, Any], item: dict[str, Any], args: argparse.Namespace) -> bool:
    requires_gpu_rerun = bool(item.get("requires_gpu_rerun"))
    if args.backend == "sd3" and entry["key"] in SD3_REUSE_OLD_KEYS and not requires_gpu_rerun:
        return False
    if args.backend == "flux" and entry["key"] in FLUX_REUSE_OLD_KEYS and not requires_gpu_rerun:
        return False
    if args.keys and entry["key"] not in set(args.keys.split(",")):
        return False
    if args.priority and item["priority"] != args.priority:
        return False
    if args.category and item["category"] != args.category:
        return False
    if args.gpu_only and not item.get("requires_gpu_rerun"):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sd3", "flux"), required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--registry", default=str(REPAIR_ROOT / "registry.json"))
    parser.add_argument("--priority", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--keys", default=None, help="Comma-separated key list.")
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else REPAIR_ROOT / f"manifest_allpass_{args.backend}_v1.json"
    manifest = load_json(manifest_path)
    registry = load_json(Path(args.registry))
    selected = [entry for entry in manifest if should_run(entry, registry[entry["key"]], args)]
    if args.limit > 0:
        selected = selected[: args.limit]
    print(f"selected={len(selected)} backend={args.backend}")
    for entry in selected:
        item = registry[entry["key"]]
        out = output_dir(args.backend, entry, item) / "result.png"
        if out.exists() and not args.overwrite:
            print(f"skip existing {entry['key']} {out}")
            continue
        cmd = build_cmd(args.backend, entry, item)
        print("+ " + " ".join(shlex.quote(str(part)) for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
            apply_postprocess_if_needed(args.backend, entry, item)


if __name__ == "__main__":
    main()
