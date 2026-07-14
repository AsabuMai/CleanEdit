from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.training_utils import set_seed
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

try:
    from operation_support_v3 import build_operation_support_v3, save_support_debug
    from run_provenance import flux_run_provenance
    from schedules import get_schedule_value
    from spatial_masks import build_object_contact_masks, save_mask_image, spatial_mask_stats
except ImportError:  # pragma: no cover - supports direct execution from flux/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from operation_support_v3 import build_operation_support_v3, save_support_debug
    from run_provenance import flux_run_provenance
    from schedules import get_schedule_value
    from spatial_masks import build_object_contact_masks, save_mask_image, spatial_mask_stats


@dataclass
class FluxEditResult:
    images: list[Image.Image]
    metadata: dict[str, object]
    stats: list[dict[str, object]]

try:
    from .dece_flux_adapter import FluxDeceAdapter, build_flux_dece_config
    from .flux_model_ops import (
        FluxLatentState,
        calc_cfg_v_flux,
        decode_flux_latents_to_unit_tensor,
        decode_flux_latents,
        encode_flux_unit_image_tensor,
        encode_flux_image,
        encode_flux_prompt,
        extract_flux_prompt_attention_map,
        flux_changed_words,
        flux_content_edit_words,
        flux_token_indices_for_words,
        invert_source_flux,
        predict_x0_from_linear_rf_path,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from dece_flux_adapter import FluxDeceAdapter, build_flux_dece_config  # type: ignore
    from flux_model_ops import (  # type: ignore
        FluxLatentState,
        calc_cfg_v_flux,
        decode_flux_latents_to_unit_tensor,
        decode_flux_latents,
        encode_flux_unit_image_tensor,
        encode_flux_image,
        encode_flux_prompt,
        extract_flux_prompt_attention_map,
        flux_changed_words,
        flux_content_edit_words,
        flux_token_indices_for_words,
        invert_source_flux,
        predict_x0_from_linear_rf_path,
    )


def _dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return table[name.lower()]


def _norm_t(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    if float(t.detach().max().item()) > 1.0:
        t = t / 1000.0
    return t


def _packed_to_map(packed: torch.Tensor, packed_h: int, packed_w: int) -> torch.Tensor:
    return packed.transpose(1, 2).reshape(packed.shape[0], packed.shape[2], packed_h, packed_w)


def _map_to_packed(mask: torch.Tensor) -> torch.Tensor:
    return mask.flatten(2).transpose(1, 2)


def _packed_gate_to_map(gate: torch.Tensor | None, packed_h: int, packed_w: int) -> torch.Tensor | None:
    if gate is None:
        return None
    if gate.ndim == 4:
        return gate
    return gate.transpose(1, 2).reshape(gate.shape[0], gate.shape[2], packed_h, packed_w)


def _packed_masked_rms(value: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    value = value.float()
    if gate is None:
        return value.square().mean().sqrt()
    gate = gate.float().to(device=value.device)
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    if gate.shape[-1] == 1 and value.shape[-1] != 1:
        gate = gate.expand(*value.shape[:-1], value.shape[-1])
    denom = gate.sum().clamp_min(1e-8)
    return ((value.square() * gate).sum() / denom).sqrt()


def _apply_packed_gate(value: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return value
    gate = gate.to(device=value.device, dtype=value.dtype)
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    if gate.shape[-1] == 1 and value.shape[-1] != 1:
        gate = gate.expand(*value.shape[:-1], value.shape[-1])
    return value * gate


def _load_mask_map(mask_path: str, packed_h: int, packed_w: int, device: torch.device) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L").resize((packed_w, packed_h), Image.Resampling.BILINEAR)
    data = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
    data = data.view(packed_h, packed_w).float() / 255.0
    return data.view(1, 1, packed_h, packed_w).to(device=device)


def _parse_word_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    words = [item.strip().lower() for item in value.replace(",", " ").split() if item.strip()]
    return words or None


_COLOR_TABLE = {
    "black": (0.02, 0.02, 0.02),
    "white": (0.95, 0.95, 0.95),
    "red": (0.78, 0.04, 0.03),
    "orange": (0.95, 0.34, 0.04),
    "blue": (0.03, 0.16, 0.78),
    "deep_blue": (0.02, 0.08, 0.45),
    "green": (0.05, 0.45, 0.12),
    "yellow": (0.95, 0.78, 0.05),
    "gold": (0.95, 0.68, 0.08),
}


def _parse_rgb(value: str | None) -> tuple[float, float, float] | None:
    if not value:
        return None
    key = value.strip().lower().replace(" ", "_")
    if key in _COLOR_TABLE:
        return _COLOR_TABLE[key]
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if len(parts) == 3:
        rgb = tuple(float(part) for part in parts)
        if max(rgb) > 1.0:
            rgb = tuple(channel / 255.0 for channel in rgb)
        return tuple(max(0.0, min(1.0, channel)) for channel in rgb)  # type: ignore[return-value]
    return None


def _component_with_overlap(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    ref_bool = reference.astype(bool)
    visited = np.zeros(mask_bool.shape, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    best_score = -1
    height, width = mask_bool.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask_bool[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            overlap = 0
            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                if ref_bool[cy, cx]:
                    overlap += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and mask_bool[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            score = overlap if overlap > 0 else -len(pixels)
            if score > best_score:
                best_score = score
                best_pixels = pixels
    out = np.zeros(mask_bool.shape, dtype=np.float32)
    for y, x in best_pixels:
        out[y, x] = 1.0
    return out


def _mask_bbox(mask: np.ndarray, threshold: float = 0.1) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask > threshold)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _star_points(cx: float, cy: float, outer: float, inner: float, rotation: float = -np.pi / 2.0) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = rotation + i * np.pi / 5.0
        pts.append((cx + radius * float(np.cos(angle)), cy + radius * float(np.sin(angle))))
    return pts


def _draw_polygon_mask(size: tuple[int, int], points: list[tuple[float, float]], blur: float = 1.2) -> np.ndarray:
    mask_img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(points, fill=255)
    if blur > 0.0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(float(blur)))
    return np.asarray(mask_img).astype(np.float32) / 255.0


def _infer_recolor_target(prompt: str, explicit: str | None) -> tuple[float, float, float] | None:
    parsed = _parse_rgb(explicit)
    if parsed is not None:
        return parsed
    lower = prompt.lower()
    for name, rgb in _COLOR_TABLE.items():
        if name.replace("_", " ") in lower or name in lower:
            return rgb
    return None


def _packed_gate_to_image(gate: torch.Tensor, packed_h: int, packed_w: int, height: int, width: int) -> torch.Tensor:
    gate_map = gate.transpose(1, 2).reshape(gate.shape[0], gate.shape[2], packed_h, packed_w)
    return F.interpolate(gate_map.float(), size=(height, width), mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def _build_flux_support(
    args,
    state: FluxLatentState,
    x_t: torch.Tensor,
    t: torch.Tensor,
    v_src: torch.Tensor,
    v_tar: torch.Tensor,
    attention_map: torch.Tensor | None = None,
    host_attention_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    if attention_map is None and not args.semantic_base_mask:
        raise ValueError("dece_rf_flux currently requires --support-mask/--semantic-base-mask")
    packed_h = state.latent_height // 2
    packed_w = state.latent_width // 2
    external_mask = (
        _load_mask_map(args.semantic_base_mask, packed_h, packed_w, x_t.device)
        if args.semantic_base_mask
        else None
    )
    if attention_map is None:
        attention_map = external_mask
    grounding = external_mask if args.support_external_mask_role == "grounding" else None
    if attention_map is None:
        raise ValueError("operation support v3 requires an attention map")
    x_map = _packed_to_map(x_t, packed_h, packed_w)
    src_map = _packed_to_map(v_src, packed_h, packed_w)
    tar_map = _packed_to_map(v_tar, packed_h, packed_w)
    support = None
    if args.support_control_mode == "fixed":
        if external_mask is None:
            raise ValueError("--support-control-mode fixed requires --support-mask/--semantic-base-mask")
        edit_map = external_mask.float().to(device=x_t.device).clamp(0.0, 1.0)
        core_map = edit_map
        fixed_attention_core = False
        if args.fixed_core_from_attention and args.use_flux_attention_support and attention_map is not None:
            attention_in_edit = attention_map.float().to(device=x_t.device).clamp(0.0, 1.0) * edit_map
            valid = attention_in_edit[edit_map > 0.05]
            if valid.numel() > 0 and float(valid.max().detach().item()) > 1e-6:
                q = max(0.0, min(1.0, float(args.fixed_core_attention_percentile) / 100.0))
                threshold = torch.quantile(valid.float(), q)
                core_map = (attention_in_edit >= threshold).float() * edit_map
                core_map = torch.maximum(core_map, 0.10 * edit_map).clamp(0.0, 1.0)
                fixed_attention_core = True
        support_stats_base: dict[str, object] = {
            "support_mode": "fixed_mask",
            "support_score": "fixed_mask",
            "support_edit_operation": args.edit_operation,
            "support_relation": args.support_relation,
            "support_has_grounding": int(external_mask is not None),
            "support_has_relation": 0,
            "support_fixed_core_from_attention": int(fixed_attention_core),
            "support_fixed_core_attention_percentile": float(args.fixed_core_attention_percentile),
        }
    else:
        support = build_operation_support_v3(
            attention_map=attention_map,
            x_t=x_map,
            t=_norm_t(t),
            source_velocity=src_map,
            target_velocity=tar_map,
            host_attention_map=host_attention_map,
            grounding_mask=grounding,
            edit_operation=args.edit_operation,
            relation=args.support_relation,
            candidate=args.support_candidate,
            top_percentile=args.support_top_percentile,
            min_area_ratio=args.support_min_area_ratio,
            max_area_ratio=args.support_max_area_ratio,
            keep_components=args.support_keep_components,
            dilate_radius=args.support_dilate_radius,
            blur_kernel=args.support_blur_kernel,
        )
        edit_map = support.edit_mask.float().to(device=x_t.device)
        core_map = support.core_mask.float().to(device=x_t.device)
        support_stats_base = dict(support.stats)
    contact_map = None
    edge_map = None
    if args.mask_layering_mode == "object_contact":
        edit_map, core_map, contact_map, preserve_map, edge_map = build_object_contact_masks(
            edit_mask=edit_map,
            core_mask=core_map,
            structure_reference=x_map,
            object_threshold=args.mask_object_threshold,
            contact_dilate_kernel=args.mask_contact_dilate_kernel,
            contact_scale=args.mask_contact_scale,
            contact_edge_threshold=args.mask_contact_edge_threshold,
            contact_edge_protect_scale=args.mask_contact_edge_protect_scale,
        )
    elif args.mask_layering_mode == "none":
        core_map = torch.minimum(core_map, edit_map)
        preserve_map = (1.0 - edit_map).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported mask_layering_mode: {args.mask_layering_mode}")
    if args.mask_output_dir:
        os.makedirs(args.mask_output_dir, exist_ok=True)
        if support is not None:
            save_support_debug(support, args.mask_output_dir)
        save_mask_image(edit_map, os.path.join(args.mask_output_dir, "subject_final.png"))
        save_mask_image(core_map, os.path.join(args.mask_output_dir, "core_final.png"))
        save_mask_image(preserve_map, os.path.join(args.mask_output_dir, "preserve_final.png"))
        if contact_map is not None:
            save_mask_image(contact_map, os.path.join(args.mask_output_dir, "contact_final.png"))
        if edge_map is not None:
            save_mask_image(edge_map, os.path.join(args.mask_output_dir, "structure_edge.png"))
    edit_gate = _map_to_packed(edit_map).to(device=x_t.device, dtype=torch.float32)
    core_gate = _map_to_packed(core_map).to(device=x_t.device, dtype=torch.float32)
    preserve_gate = _map_to_packed(preserve_map).to(device=x_t.device, dtype=torch.float32)
    stats = {
        **support_stats_base,
        **spatial_mask_stats(edit_map, prefix="mask"),
        **spatial_mask_stats(core_map, prefix="core_mask"),
        **spatial_mask_stats(preserve_map, prefix="preserve_mask"),
    }
    return edit_gate, core_gate, preserve_gate, stats


def _ensure_parent(path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal FLUX CleanEdit transfer runner")
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-prompt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--run-purpose",
        default="diagnostic",
        choices=["diagnostic", "evaluation", "paper"],
        help="Declares intended use; eligibility is derived from the effective configuration.",
    )
    parser.add_argument(
        "--comparison-protocol",
        default="single_pass",
        choices=["single_pass", "diagnostic_multi_pass"],
    )
    parser.add_argument(
        "--method",
        default="base_only_flux",
        choices=["base_only_flux", "direct_target_flux", "dece_rf_flux", "dece_rf_flux_minimal"],
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--torch-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--true-cfg",
        action="store_true",
        help="Use true classifier-free guidance (cond/uncond extrapolation against a "
        "negative prompt) for the CleanEdit velocities, holding the distilled FLUX "
        "guidance embedding at --distilled-guidance. Faithful to the SD3 CFG mechanism.",
    )
    parser.add_argument(
        "--distilled-guidance",
        type=float,
        default=1.0,
        help="Distilled FLUX guidance embedding value held fixed when --true-cfg is on.",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--src-guidance-scale", type=float, default=None)
    parser.add_argument("--tar-guidance-scale", type=float, default=None)
    parser.add_argument("--base-guidance-scale", type=float, default=None)
    parser.add_argument("--inversion-guidance-scale", type=float, default=None)
    parser.add_argument("--edit-src-cfg-scale", type=float, default=None)
    parser.add_argument("--rec-guidance-scale", type=float, default=0.35)
    parser.add_argument("--struct-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-hedit-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-guidance-scale", type=float, default=0.65)
    parser.add_argument("--edit-region-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-target-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-source-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-core-scale", type=float, default=1.35)
    parser.add_argument("--edit-subject-scale", type=float, default=0.35)
    parser.add_argument("--edit-local-target-prompt", default=None)
    parser.add_argument("--edit-local-target-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-local-target-cfg-scale", type=float, default=None)
    parser.add_argument("--recolor-target", default=None)
    parser.add_argument("--recolor-clean-projection-scale", type=float, default=0.0)
    parser.add_argument("--recolor-clean-projection-alpha", type=float, default=0.85)
    parser.add_argument("--recolor-preserve-luma-scale", type=float, default=0.65)
    # texture-preserving recolor: weight of the source-textured recolor target vs the
    # flat luma-matched color. 1.0 = fully texture-preserving (keeps spokes/scales/shading),
    # 0.0 = legacy flat-color projection (uniform hue, "painted on" look).
    parser.add_argument("--recolor-texture-preserve-scale", type=float, default=0.85)
    parser.add_argument("--removal-controller-mode", default="none", choices=["none", "clean_fill"])
    parser.add_argument("--removal-fill-scale", type=float, default=0.0)
    parser.add_argument("--removal-suppression-scale", type=float, default=0.0)
    parser.add_argument("--removal-ring-rec-scale", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--alpha-schedule", default="constant")
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--beta-schedule", default="constant")
    parser.add_argument("--linear-path-t-min", type=float, default=0.05)
    parser.add_argument("--rec-stop-timestep", type=float, default=0.08)
    parser.add_argument("--adaptive-clean-control", action="store_true")
    parser.add_argument("--adaptive-edit-target-progress", type=float, default=0.0)
    parser.add_argument("--adaptive-rmsgap-mode", default="legacy", choices=["legacy", "normgate"])
    parser.add_argument("--adaptive-edit-target-rms", type=float, default=0.42)
    parser.add_argument("--adaptive-rmsgap-dead-zone", type=float, default=0.0)
    parser.add_argument("--adaptive-rmsgap-preserve-gate-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-preserve-drift-budget", type=float, default=0.18)
    parser.add_argument("--adaptive-edit-gain", type=float, default=2.0)
    parser.add_argument("--clean-diagnostics-edit-mask", default=None)
    parser.add_argument("--clean-diagnostics-preserve-mask", default=None)
    parser.add_argument("--final-blend-mask", default=None)
    parser.add_argument("--keep-text-encoders-cpu", action="store_true")
    parser.add_argument("--log-clean-diagnostics", action="store_true")
    parser.add_argument("--prompt-encode-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--adaptive-component-control", action="store_true")
    parser.add_argument("--adaptive-component-threshold", type=float, default=0.5)
    parser.add_argument("--adaptive-component-min-pixels", type=int, default=4)
    parser.add_argument("--adaptive-preserve-gain", type=float, default=2.5)
    parser.add_argument("--adaptive-edit-weight-min", type=float, default=0.85)
    parser.add_argument("--adaptive-edit-weight-max", type=float, default=1.55)
    parser.add_argument("--adaptive-preserve-weight-min", type=float, default=1.0)
    parser.add_argument("--adaptive-preserve-weight-max", type=float, default=1.65)
    parser.add_argument("--adaptive-projection-scale", type=float, default=0.65)
    parser.add_argument("--adaptive-preserve-clean-correction-scale", type=float, default=0.5)
    parser.add_argument("--trajectory-preserve-scale", type=float, default=0.12)
    parser.add_argument("--trajectory-subject-preserve-scale", type=float, default=0.0)
    parser.add_argument("--region-target-transport-scale", type=float, default=0.0)
    parser.add_argument("--source-attachment-release-scale", type=float, default=0.0)
    parser.add_argument("--source-attachment-release-stop-t", type=float, default=0.35)
    parser.add_argument("--source-attachment-release-full-t", type=float, default=0.65)
    parser.add_argument("--region-target-outside-lock-scale", type=float, default=0.0)
    parser.add_argument("--minimal-core-scale", type=float, default=1.0)
    parser.add_argument("--minimal-ring-scale", type=float, default=0.35)
    parser.add_argument("--minimal-outside-lock-scale", type=float, default=1.0)
    parser.add_argument("--minimal-recolor-scale", type=float, default=0.0)
    parser.add_argument("--minimal-n-avg", type=int, default=4)
    parser.add_argument("--minimal-edit-scale", type=float, default=1.0)
    parser.add_argument(
        "--final-postprocess-mode",
        default="mask_blend",
        choices=["none", "mask_blend", "debug_visual"],
        help="Final compositing policy. debug_visual enables legacy task-specific drawing overlays.",
    )
    parser.add_argument("--final-mask-blend-scale", type=float, default=1.0)
    parser.add_argument(
        "--final-mask-alpha-gamma",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to the final edit mask before source/result compositing. "
            "Values below 1 strengthen soft edit-mask cores without changing zero-alpha background."
        ),
    )
    parser.add_argument("--final-recolor-blend-scale", type=float, default=0.0)
    parser.add_argument("--final-knit-texture-scale", type=float, default=0.0)
    parser.add_argument("--final-knit-source-blend", type=float, default=0.0)
    parser.add_argument("--final-mask-dilate", type=int, default=0)
    parser.add_argument("--final-source-color-mask", default=None)
    parser.add_argument("--recolor-full-mask", action="store_true")
    parser.add_argument("--final-object-overlay", default=None)
    parser.add_argument("--model-offload", action="store_true")
    parser.add_argument(
        "--sequential-offload",
        action="store_true",
        help="Use sequential CPU offload (low peak GPU memory, slower). Preferred on a shared GPU.",
    )
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--semantic-base-mask", "--support-mask", dest="semantic_base_mask", default=None)
    parser.add_argument("--support-control-mode", default="operation", choices=["operation", "fixed"])
    parser.add_argument(
        "--support-mode",
        default=None,
        choices=["operation_v3", "operation", "fixed", "generic"],
        help="SD3-compatible support selector mapped onto the FLUX support adapter.",
    )
    parser.add_argument(
        "--object-mask-provider",
        default=None,
        help="SD3-compatible mask provider name recorded for interface parity.",
    )
    parser.add_argument("--edit-operation", default="auto")
    parser.add_argument("--support-relation", default="auto")
    parser.add_argument("--support-score", default=None)
    parser.add_argument("--support-candidate", default="operation_default")
    parser.add_argument(
        "--support-external-mask-role",
        default="attention",
        choices=["attention", "grounding"],
        help="Whether --semantic-base-mask is a proxy attention/edit map or a grounding/host mask.",
    )
    parser.add_argument("--support-top-percentile", type=float, default=90.0)
    parser.add_argument("--support-min-area-ratio", type=float, default=0.02)
    parser.add_argument("--support-max-area-ratio", type=float, default=0.30)
    parser.add_argument("--support-keep-components", type=int, default=2)
    parser.add_argument("--support-dilate-radius", type=int, default=5)
    parser.add_argument("--support-blur-kernel", type=int, default=5)
    parser.add_argument("--fixed-core-from-attention", action="store_true")
    parser.add_argument("--fixed-core-attention-percentile", type=float, default=82.0)
    parser.add_argument("--use-flux-attention-support", action="store_true")
    parser.add_argument("--new-tokens", default=None)
    parser.add_argument("--host-tokens", default=None)
    parser.add_argument("--removed-tokens", default=None)
    parser.add_argument("--grounding-method", default=None)
    parser.add_argument("--save-support-debug", action="store_true")
    parser.add_argument("--support-debug-only", action="store_true")
    parser.add_argument("--flux-attention-self-weight", type=float, default=0.0)
    parser.add_argument("--flux-attention-layer-stride", type=int, default=1)
    parser.add_argument("--mask-layering-mode", "--mask-layering", dest="mask_layering_mode", default="object_contact", choices=["object_contact", "none"])
    parser.add_argument("--mask-object-threshold", type=float, default=0.45)
    parser.add_argument("--mask-contact-dilate-kernel", type=int, default=7)
    parser.add_argument("--mask-contact-scale", type=float, default=0.25)
    parser.add_argument("--mask-contact-edge-threshold", type=float, default=0.55)
    parser.add_argument("--mask-contact-edge-protect-scale", type=float, default=0.75)
    parser.add_argument("--mask-output-dir", default=None)
    parser.add_argument("--mask-blend", action="store_true", default=False)
    parser.add_argument("--mask-blend-mode", default="subject", choices=["subject", "core"])
    parser.add_argument("--max-sequence-length", type=int, default=512)
    return parser


def _normalize_sd3_interface_args(args) -> dict[str, object]:
    """Map SD3 CLI spellings onto the FLUX adapter without changing edit math."""
    interface: dict[str, object] = {
        "support_mode_arg": args.support_mode,
        "object_mask_provider_arg": args.object_mask_provider,
        "support_score_arg": args.support_score,
        "mask_blend_arg": bool(args.mask_blend),
        "mask_blend_mode_arg": args.mask_blend_mode,
    }
    if args.support_mode in {"operation_v3", "operation"}:
        args.support_control_mode = "operation"
    elif args.support_mode in {"fixed", "generic"}:
        args.support_control_mode = "fixed"
    if args.object_mask_provider == "operation_support_v3":
        args.support_control_mode = "operation"
    elif args.object_mask_provider in {"semantic", "semantic_velocity"}:
        args.support_control_mode = "fixed"
    if args.support_score and args.support_candidate == "operation_default":
        args.support_candidate = args.support_score
    if args.mask_blend:
        args.final_postprocess_mode = "mask_blend"
    return interface


def load_flux_pipeline(args, device: torch.device):
    dtype = _dtype_from_name(args.torch_dtype)
    print(
        f"[flux] loading pipeline model_id={args.model_id} cache_dir={args.cache_dir} "
        f"local_files_only={args.local_files_only}",
        flush=True,
    )
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if getattr(args, "sequential_offload", False) and hasattr(pipe, "enable_sequential_cpu_offload"):
        print("[flux] enabling sequential cpu offload (low peak GPU memory)", flush=True)
        pipe.enable_sequential_cpu_offload()
    elif args.model_offload and hasattr(pipe, "enable_model_cpu_offload"):
        print("[flux] enabling model cpu offload", flush=True)
        pipe.enable_model_cpu_offload()
    else:
        print(f"[flux] moving pipeline to {device}", flush=True)
        pipe = pipe.to(device)
    print("[flux] pipeline ready", flush=True)
    return pipe


def run_flux_edit(args) -> tuple[list[Image.Image], dict[str, object], list[dict[str, object]]]:
    if args.adaptive_component_control:
        args.adaptive_clean_control = True
    sd3_interface = _normalize_sd3_interface_args(args)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = load_flux_pipeline(args, device)
    print("[flux] setting scheduler timesteps", flush=True)
    scheduler = pipe.scheduler

    src_guidance = args.src_guidance_scale if args.src_guidance_scale is not None else args.guidance_scale
    tar_guidance = args.tar_guidance_scale if args.tar_guidance_scale is not None else args.guidance_scale
    base_guidance = args.base_guidance_scale if args.base_guidance_scale is not None else src_guidance
    inversion_guidance = args.inversion_guidance_scale if args.inversion_guidance_scale is not None else src_guidance
    edit_src_guidance = args.edit_src_cfg_scale if args.edit_src_cfg_scale is not None else base_guidance
    edit_local_target_cfg = (
        args.edit_local_target_cfg_scale if args.edit_local_target_cfg_scale is not None else tar_guidance
    )
    alpha_max = args.alpha_max if args.alpha_max is not None else args.rec_guidance_scale
    beta_max = args.beta_max if args.beta_max is not None else 1.0

    src_prompt = encode_flux_prompt(
        pipe,
        args.source_prompt,
        device=device,
        max_sequence_length=args.max_sequence_length,
    )
    print("[flux] encoded source prompt", flush=True)
    tar_prompt = encode_flux_prompt(
        pipe,
        args.prompt,
        device=device,
        max_sequence_length=args.max_sequence_length,
    )
    print("[flux] encoded target prompt", flush=True)
    neg_prompt = None
    if args.true_cfg:
        neg_prompt = encode_flux_prompt(
            pipe,
            args.negative_prompt,
            device=device,
            max_sequence_length=args.max_sequence_length,
        )
        print(
            f"[flux] true CFG enabled: negative prompt encoded, "
            f"distilled_guidance={args.distilled_guidance}",
            flush=True,
        )
    neg_embeds = neg_prompt["prompt_embeds"] if neg_prompt is not None else None
    neg_pooled = neg_prompt["pooled_prompt_embeds"] if neg_prompt is not None else None
    distilled_g = args.distilled_guidance if args.true_cfg else None
    local_target_prompt = None
    if args.edit_local_target_prompt:
        local_target_prompt = encode_flux_prompt(
            pipe,
            args.edit_local_target_prompt,
            device=device,
            max_sequence_length=args.max_sequence_length,
        )
        print("[flux] encoded local target prompt", flush=True)
    state = encode_flux_image(
        pipe,
        args.image,
        device=device,
        max_image_size=args.max_image_size,
        dtype=_dtype_from_name(args.torch_dtype),
    )
    print(
        f"[flux] encoded image latent shape={tuple(state.latents.shape)} "
        f"image={state.image_width}x{state.image_height}",
        flush=True,
    )
    image_seq_len = state.latents.shape[1]
    dece_adapter = FluxDeceAdapter(state.latent_height // 2, state.latent_width // 2)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    sigmas = np.linspace(1.0, 1.0 / args.num_inference_steps, args.num_inference_steps)
    scheduler.set_timesteps(
        args.num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )
    timesteps = scheduler.timesteps
    T_steps = len(timesteps)
    print(f"[flux] timesteps ready steps={T_steps} mu={mu:.4f}", flush=True)
    x_src = state.latents
    use_minimal_flux = args.method == "dece_rf_flux_minimal"
    need_source_trajectory = args.method == "dece_rf_flux" and (
        args.trajectory_preserve_scale > 0.0
        or args.trajectory_subject_preserve_scale > 0.0
        or args.region_target_outside_lock_scale > 0.0
    )
    if use_minimal_flux:
        z_t = x_src.clone()
        source_trajectory_by_timestep = {}
        print("[flux] minimal mode starts from packed source latent", flush=True)
    else:
        print("[flux] running source inversion", flush=True)
        inversion_result = invert_source_flux(
            pipe=pipe,
            x_src=x_src,
            prompt_embeds=src_prompt["prompt_embeds"],
            pooled_prompt_embeds=src_prompt["pooled_prompt_embeds"],
            text_ids=src_prompt["text_ids"],
            latent_image_ids=state.latent_image_ids,
            guidance_scale=inversion_guidance,
            timesteps=timesteps,
            sigmas=scheduler.sigmas,
            T_steps=T_steps,
            n_max=args.n_max,
            return_trajectory=need_source_trajectory,
        )
        if need_source_trajectory:
            z_t, source_trajectory_by_timestep = inversion_result
        else:
            z_t = inversion_result
            source_trajectory_by_timestep = {}
        print("[flux] source inversion done", flush=True)

    stats: list[dict[str, object]] = []
    support_stats: dict[str, object] = {}
    edit_gate = None
    core_gate = None
    preserve_gate = None
    recolor_projection_target = None
    recolor_projection_gate = None
    recolor_rgb = None
    target_feature_proxy_map = None
    source_feature_proxy_map = None
    target_feature_map_source = None
    source_feature_map_source = None
    if args.method in {"dece_rf_flux", "dece_rf_flux_minimal"}:
        mid_index = len(timesteps) // 2
        t_mid = timesteps[mid_index]
        sigma_mid = scheduler.sigmas[mid_index].to(device=device, dtype=torch.float32)
        packed_h = state.latent_height // 2
        packed_w = state.latent_width // 2
        with torch.no_grad():
            v_src_support = calc_cfg_v_flux(
                pipe=pipe,
                latents=x_src,
                prompt_embeds=src_prompt["prompt_embeds"],
                pooled_prompt_embeds=src_prompt["pooled_prompt_embeds"],
                text_ids=src_prompt["text_ids"],
                latent_image_ids=state.latent_image_ids,
                guidance_scale=base_guidance,
                t=t_mid,
            ).to(torch.float32)
            v_tar_support = calc_cfg_v_flux(
                pipe=pipe,
                latents=x_src,
                prompt_embeds=tar_prompt["prompt_embeds"],
                pooled_prompt_embeds=tar_prompt["pooled_prompt_embeds"],
                text_ids=tar_prompt["text_ids"],
                latent_image_ids=state.latent_image_ids,
                guidance_scale=tar_guidance,
                t=t_mid,
            ).to(torch.float32)
        attention_map = None
        host_attention_map = None
        if args.use_flux_attention_support:
            src_changed, tar_changed = flux_changed_words(args.source_prompt, args.prompt)
            new_words = _parse_word_list(args.new_tokens)
            host_words = _parse_word_list(args.host_tokens)
            if new_words is None:
                new_words = flux_content_edit_words(tar_changed)
            if host_words is None:
                host_words = flux_content_edit_words(src_changed)
            new_token_indices = flux_token_indices_for_words(
                pipe,
                args.prompt,
                new_words,
                max_sequence_length=args.max_sequence_length,
            )
            host_token_indices = flux_token_indices_for_words(
                pipe,
                args.source_prompt,
                host_words,
                max_sequence_length=args.max_sequence_length,
            )
            n_blocks = len(pipe.transformer.transformer_blocks)
            lo = n_blocks // 4
            hi = max(lo + 1, (3 * n_blocks) // 4)
            stride = max(1, int(args.flux_attention_layer_stride))
            layer_indices = list(range(lo, hi, stride))
            attention_map = extract_flux_prompt_attention_map(
                pipe=pipe,
                latents=x_src,
                prompt_embeds=tar_prompt["prompt_embeds"],
                pooled_prompt_embeds=tar_prompt["pooled_prompt_embeds"],
                text_ids=tar_prompt["text_ids"],
                latent_image_ids=state.latent_image_ids,
                t=t_mid,
                guidance_scale=tar_guidance,
                packed_h=packed_h,
                packed_w=packed_w,
                token_indices=new_token_indices,
                layer_indices=layer_indices,
                self_weight=args.flux_attention_self_weight,
            ).to(device=x_src.device, dtype=torch.float32)
            if host_token_indices:
                host_attention_map = extract_flux_prompt_attention_map(
                    pipe=pipe,
                    latents=x_src,
                    prompt_embeds=src_prompt["prompt_embeds"],
                    pooled_prompt_embeds=src_prompt["pooled_prompt_embeds"],
                    text_ids=src_prompt["text_ids"],
                    latent_image_ids=state.latent_image_ids,
                    t=t_mid,
                    guidance_scale=base_guidance,
                    packed_h=packed_h,
                    packed_w=packed_w,
                    token_indices=host_token_indices,
                    layer_indices=layer_indices,
                    self_weight=args.flux_attention_self_weight,
                ).to(device=x_src.device, dtype=torch.float32)
            print(
                "[flux] attention support "
                f"new_words={new_words} new_tokens={new_token_indices} "
                f"host_words={host_words} host_tokens={host_token_indices} "
                f"layers={layer_indices}",
                flush=True,
            )
        edit_gate, core_gate, preserve_gate, support_stats = _build_flux_support(
            args,
            state,
            x_src,
            sigma_mid,
            v_src_support,
            v_tar_support,
            attention_map=attention_map,
            host_attention_map=host_attention_map,
        )
        print(
            "[flux] support "
            f"score={support_stats.get('support_score')} "
            f"area={support_stats.get('mask_area_ratio')}",
            flush=True,
        )
        target_feature_proxy_map = attention_map.detach() if attention_map is not None else None
        source_feature_proxy_map = host_attention_map.detach() if host_attention_map is not None else None
        target_feature_map_source = "flux_attention" if target_feature_proxy_map is not None else None
        source_feature_map_source = "flux_attention" if source_feature_proxy_map is not None else None
        recolor_rgb = _infer_recolor_target(args.prompt, args.recolor_target)
        if args.recolor_clean_projection_scale > 0.0 and recolor_rgb is not None:
            source_image = decode_flux_latents_to_unit_tensor(pipe, state)
            edit_image_gate = _packed_gate_to_image(
                edit_gate,
                state.latent_height // 2,
                state.latent_width // 2,
                state.image_height,
                state.image_width,
            ).to(device=source_image.device, dtype=source_image.dtype)
            target_rgb = torch.tensor(recolor_rgb, device=source_image.device, dtype=source_image.dtype).view(1, 3, 1, 1)
            luma = source_image.mean(dim=1, keepdim=True)
            target_luma = target_rgb.mean(dim=1, keepdim=True).clamp_min(1e-6)
            luma_matched = (target_rgb * (luma / target_luma)).clamp(0.0, 1.0)
            # legacy flat color: spatially-uniform hue (luma-modulated). Kept only as a
            # small fallback so dull sources can still reach a saturated target.
            flat_color = (
                float(args.recolor_preserve_luma_scale) * luma_matched
                + (1.0 - float(args.recolor_preserve_luma_scale)) * target_rgb
            ).clamp(0.0, 1.0)
            # texture-preserving recolor: a per-channel gain recolors the source while keeping
            # every pixel's detail (gain = target / source-mean), then it matches the source
            # luminance so all shading/texture (spokes, scales, folds) survives and only the
            # hue moves. This is what makes the result look recolored instead of "painted on".
            region_mask = (edit_image_gate > 0.05).to(source_image.dtype)
            denom = region_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            src_mean = (source_image * region_mask).sum(dim=(2, 3), keepdim=True) / denom
            gain = target_rgb / src_mean.clamp_min(1e-3)
            recolored = (source_image * gain).clamp(0.0, 1.0)
            rec_luma = recolored.mean(dim=1, keepdim=True).clamp_min(1e-6)
            recolored = (recolored * (luma / rec_luma)).clamp(0.0, 1.0)
            tex_w = float(args.recolor_texture_preserve_scale)
            target_color = (tex_w * recolored + (1.0 - tex_w) * flat_color).clamp(0.0, 1.0)
            alpha = (float(args.recolor_clean_projection_alpha) * edit_image_gate).clamp(0.0, 1.0)
            projected_image = (source_image * (1.0 - alpha) + target_color * alpha).clamp(0.0, 1.0)
            recolor_state = encode_flux_unit_image_tensor(
                pipe,
                projected_image,
                image_height=state.image_height,
                image_width=state.image_width,
                latent_image_ids=state.latent_image_ids,
                dtype=_dtype_from_name(args.torch_dtype),
            )
            recolor_projection_target = recolor_state.latents.to(device=x_src.device, dtype=torch.float32)
            recolor_projection_gate = edit_gate.to(device=x_src.device, dtype=torch.float32)
            print(f"[flux] recolor projection target rgb={recolor_rgb}", flush=True)
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        if T_steps - i > args.n_max:
            continue
        sigma_i = scheduler.sigmas[i].to(device=device, dtype=torch.float32)
        sigma_next = scheduler.sigmas[i + 1].to(device=device, dtype=torch.float32)
        alpha_t = get_schedule_value(args.alpha_schedule, i, len(timesteps), alpha_max)
        if args.rec_stop_timestep > 0.0 and float(sigma_i.detach().item()) < args.rec_stop_timestep:
            alpha_t = 0.0
        beta_t = get_schedule_value(args.beta_schedule, i, len(timesteps), beta_max)
        latents_dtype = z_t.dtype
        with torch.no_grad():
            v_src = calc_cfg_v_flux(
                pipe=pipe,
                latents=z_t.to(latents_dtype),
                prompt_embeds=src_prompt["prompt_embeds"],
                pooled_prompt_embeds=src_prompt["pooled_prompt_embeds"],
                text_ids=src_prompt["text_ids"],
                latent_image_ids=state.latent_image_ids,
                guidance_scale=base_guidance,
                t=t,
                negative_prompt_embeds=neg_embeds,
                negative_pooled_prompt_embeds=neg_pooled,
                distilled_guidance=distilled_g,
            ).to(torch.float32)
            v_tar = calc_cfg_v_flux(
                pipe=pipe,
                latents=z_t.to(latents_dtype),
                prompt_embeds=tar_prompt["prompt_embeds"],
                pooled_prompt_embeds=tar_prompt["pooled_prompt_embeds"],
                text_ids=tar_prompt["text_ids"],
                latent_image_ids=state.latent_image_ids,
                guidance_scale=tar_guidance,
                t=t,
                negative_prompt_embeds=neg_embeds,
                negative_pooled_prompt_embeds=neg_pooled,
                distilled_guidance=distilled_g,
            ).to(torch.float32)
        z_t = z_t.to(torch.float32)
        x0_src_step = predict_x0_from_linear_rf_path(z_t, v_src, sigma_i)
        x0_tar = predict_x0_from_linear_rf_path(z_t, v_tar, sigma_i)
        recolor_clean_projection_norm = 0.0
        removal_controller_norm = 0.0
        removal_fill_norm = 0.0
        removal_suppression_norm = 0.0
        removal_ring_rec_norm = 0.0
        region_target_transport_norm = 0.0
        region_target_transport_core_beta = 0.0
        region_target_transport_ring_beta = 0.0
        region_target_transport_core_gamma = 0.0
        region_target_outside_lock_norm = 0.0
        region_target_outside_lock_weight = 0.0
        trajectory_preserve_norm = 0.0
        trajectory_preserve_weight = 0.0
        core_diagnostics: dict[str, float] = {}

        if args.method == "base_only_flux":
            v_total = v_src
            v_rec = torch.zeros_like(v_src)
            v_edit = torch.zeros_like(v_src)
            adaptive_edit_weight = 1.0
            adaptive_preserve_weight = 1.0
            adaptive_projection_norm = 0.0
            adaptive_preserve_clean_correction_norm = 0.0
            edit_terms_norm = {"base": 0.0, "anchor": 0.0, "region": 0.0, "target": 0.0, "source": 0.0}
        elif args.method == "direct_target_flux":
            v_total = v_tar
            v_rec = torch.zeros_like(v_src)
            v_edit = v_tar - v_src
            adaptive_edit_weight = 1.0
            adaptive_preserve_weight = 1.0
            adaptive_projection_norm = 0.0
            adaptive_preserve_clean_correction_norm = 0.0
            edit_terms_norm = {
                "base": float((v_tar - v_src).norm().item()),
                "anchor": 0.0,
                "region": 0.0,
                "target": 0.0,
                "source": 0.0,
            }
        elif use_minimal_flux:
            assert core_gate is not None
            assert edit_gate is not None
            full_gate = edit_gate.to(device=z_t.device, dtype=torch.float32).clamp(0.0, 1.0)
            core_gate_step = core_gate.to(device=z_t.device, dtype=torch.float32).clamp(0.0, 1.0)
            ring_gate = (full_gate - core_gate_step).clamp(0.0, 1.0)
            local_gate = (
                float(args.minimal_core_scale) * core_gate_step
                + float(args.minimal_ring_scale) * ring_gate
            ).clamp(0.0, 1.0)
            v_delta_avg = torch.zeros_like(x_src, dtype=torch.float32)
            n_avg = max(1, int(args.minimal_n_avg))
            for _ in range(n_avg):
                fwd_noise = torch.randn_like(x_src)
                zt_src = (1.0 - sigma_i) * x_src.to(torch.float32) + sigma_i * fwd_noise.to(torch.float32)
                zt_tar = z_t.to(torch.float32) + zt_src - x_src.to(torch.float32)
                with torch.no_grad():
                    v_src_sample = calc_cfg_v_flux(
                        pipe=pipe,
                        latents=zt_src.to(latents_dtype),
                        prompt_embeds=src_prompt["prompt_embeds"],
                        pooled_prompt_embeds=src_prompt["pooled_prompt_embeds"],
                        text_ids=src_prompt["text_ids"],
                        latent_image_ids=state.latent_image_ids,
                        guidance_scale=src_guidance,
                        t=t,
                    ).to(torch.float32)
                    v_tar_sample = calc_cfg_v_flux(
                        pipe=pipe,
                        latents=zt_tar.to(latents_dtype),
                        prompt_embeds=tar_prompt["prompt_embeds"],
                        pooled_prompt_embeds=tar_prompt["pooled_prompt_embeds"],
                        text_ids=tar_prompt["text_ids"],
                        latent_image_ids=state.latent_image_ids,
                        guidance_scale=tar_guidance,
                        t=t,
                    ).to(torch.float32)
                v_delta_avg = v_delta_avg + (v_tar_sample - v_src_sample) / float(n_avg)
            v_delta = float(args.minimal_edit_scale) * v_delta_avg * local_gate
            if recolor_projection_target is not None:
                recolor_delta = (recolor_projection_target - x_src.to(torch.float32)) * full_gate
                recolor_velocity = dece_adapter.clean_delta_to_native_velocity(
                    recolor_delta,
                    sigma_i,
                    eps=float(args.linear_path_t_min),
                )
                v_delta = v_delta + float(args.minimal_recolor_scale) * recolor_velocity
                recolor_clean_projection_norm = float((float(args.minimal_recolor_scale) * recolor_velocity).norm().item())
            v_total = v_delta
            v_rec = torch.zeros_like(v_src)
            v_edit = v_delta
            adaptive_edit_weight = 1.0
            adaptive_preserve_weight = 1.0
            adaptive_projection_norm = 0.0
            adaptive_preserve_clean_correction_norm = 0.0
            edit_terms_norm = {
                "base": float((float(args.minimal_edit_scale) * v_delta_avg * local_gate).norm().item()),
                "anchor": 0.0,
                "region": 0.0,
                "target": 0.0,
                "source": 0.0,
            }
        else:
            assert preserve_gate is not None
            assert core_gate is not None
            assert edit_gate is not None
            edit_map = dece_adapter.gate_to_map(edit_gate)
            assert edit_map is not None
            target_feature_map = target_feature_proxy_map
            source_feature_map = source_feature_proxy_map
            if target_feature_map is None and args.edit_target_guidance_scale > 0.0:
                target_feature_map = edit_map.detach()
                target_feature_map_source = target_feature_map_source or "support_mask_proxy"
            if source_feature_map is None and args.edit_source_guidance_scale > 0.0:
                source_feature_map = edit_map.detach()
                source_feature_map_source = source_feature_map_source or "support_mask_proxy"

            if edit_src_guidance == base_guidance:
                v_src_edit = v_src
            else:
                with torch.no_grad():
                    v_src_edit = calc_cfg_v_flux(
                        pipe=pipe,
                        latents=z_t.to(latents_dtype),
                        prompt_embeds=src_prompt["prompt_embeds"],
                        pooled_prompt_embeds=src_prompt["pooled_prompt_embeds"],
                        text_ids=src_prompt["text_ids"],
                        latent_image_ids=state.latent_image_ids,
                        guidance_scale=edit_src_guidance,
                        t=t,
                        negative_prompt_embeds=neg_embeds,
                        negative_pooled_prompt_embeds=neg_pooled,
                        distilled_guidance=distilled_g,
                    ).to(torch.float32)
            x0_local = None
            if (
                local_target_prompt is not None
                and args.edit_local_target_guidance_scale > 0.0
            ):
                with torch.no_grad():
                    v_local = calc_cfg_v_flux(
                        pipe=pipe,
                        latents=z_t.to(latents_dtype),
                        prompt_embeds=local_target_prompt["prompt_embeds"],
                        pooled_prompt_embeds=local_target_prompt["pooled_prompt_embeds"],
                        text_ids=local_target_prompt["text_ids"],
                        latent_image_ids=state.latent_image_ids,
                        guidance_scale=edit_local_target_cfg,
                        t=t,
                        negative_prompt_embeds=neg_embeds,
                        negative_pooled_prompt_embeds=neg_pooled,
                        distilled_guidance=distilled_g,
                    ).to(torch.float32)
                x0_local = predict_x0_from_linear_rf_path(z_t, v_local, sigma_i)
            core_out = dece_adapter.compute_step(
                build_flux_dece_config(args),
                z_t=z_t,
                x_src=x_src,
                x0_src=x0_src_step,
                x0_tar=x0_tar,
                v_src=v_src,
                v_tar=v_tar,
                v_src_edit=v_src_edit,
                t_scalar=sigma_i,
                alpha_t=alpha_t,
                beta_t=beta_t,
                edit_gate=edit_gate,
                preserve_gate=preserve_gate,
                core_gate=core_gate,
                target_feature_map=target_feature_map,
                source_feature_map=source_feature_map,
                x0_local_target=x0_local,
                recolor_projection_target=recolor_projection_target,
                recolor_projection_gate=recolor_projection_gate,
            )
            v_rec = core_out.v_rec
            v_edit = core_out.v_edit
            v_total = core_out.v_total
            core_diagnostics = core_out.diagnostics
            edit_terms_norm = core_out.edit_terms_norm
            adaptive_edit_weight = core_out.diagnostics["adaptive_edit_weight"]
            adaptive_preserve_weight = core_out.diagnostics["adaptive_preserve_weight"]
            adaptive_projection_norm = core_out.diagnostics["adaptive_projection_norm"]
            adaptive_preserve_clean_correction_norm = core_out.diagnostics[
                "adaptive_preserve_clean_correction_norm"
            ]
            recolor_clean_projection_norm = core_out.diagnostics["recolor_clean_projection_norm"]
            removal_controller_norm = core_out.diagnostics["removal_controller_norm"]
            removal_fill_norm = core_out.diagnostics["removal_fill_norm"]
            removal_suppression_norm = core_out.diagnostics["removal_suppression_norm"]
            removal_ring_rec_norm = core_out.diagnostics["removal_ring_rec_norm"]
            region_target_transport_norm = core_out.diagnostics["region_target_transport_norm"]
            region_target_transport_core_beta = core_out.diagnostics["region_target_transport_core_beta"]
            region_target_transport_ring_beta = core_out.diagnostics["region_target_transport_ring_beta"]
            region_target_transport_core_gamma = core_out.diagnostics["region_target_transport_core_gamma"]

        next_z_t = z_t + (sigma_next - sigma_i).to(torch.float32) * v_total.to(torch.float32)
        if use_minimal_flux and edit_gate is not None:
            full_gate = edit_gate.to(device=next_z_t.device, dtype=torch.float32).clamp(0.0, 1.0)
            core_gate_step = (core_gate if core_gate is not None else edit_gate).to(
                device=next_z_t.device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
            ring_gate = (full_gate - core_gate_step).clamp(0.0, 1.0)
            visible_gate = (core_gate_step + float(args.minimal_ring_scale) * ring_gate).clamp(0.0, 1.0)
            outside_gate = (1.0 - full_gate).clamp(0.0, 1.0)
            lock_gate = (float(args.minimal_outside_lock_scale) * outside_gate).clamp(0.0, 1.0)
            next_z_t = (
                visible_gate * next_z_t.to(torch.float32)
                + (1.0 - visible_gate) * z_t.to(torch.float32)
            )
            next_z_t = next_z_t + lock_gate * (x_src.to(next_z_t.device, torch.float32) - next_z_t)
            region_target_outside_lock_norm = float((lock_gate * (x_src.to(next_z_t.device, torch.float32) - next_z_t)).norm().item())
        if args.region_target_outside_lock_scale > 0.0 and edit_gate is not None:
            next_key = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else 0
            source_next = source_trajectory_by_timestep.get(next_key)
            if source_next is not None:
                t_value = float(sigma_i.detach().item())
                if t_value > 0.65:
                    outside_lock_base = 0.5
                    alpha_ring = 0.6
                elif t_value > 0.35:
                    outside_lock_base = 0.75
                    alpha_ring = 0.5
                else:
                    outside_lock_base = 0.9
                    alpha_ring = 0.3
                lock_edit_gate = edit_gate.to(device=next_z_t.device, dtype=torch.float32).clamp(0.0, 1.0)
                lock_core_gate = (core_gate if core_gate is not None else edit_gate).to(
                    device=next_z_t.device,
                    dtype=torch.float32,
                ).clamp(0.0, 1.0)
                lock_ring_gate = (lock_edit_gate - lock_core_gate).clamp(0.0, 1.0)
                outside_gate = (1.0 - lock_edit_gate).clamp(0.0, 1.0)
                region_target_outside_lock_weight = max(
                    0.0,
                    min(1.0, float(args.region_target_outside_lock_scale) * outside_lock_base),
                )
                ring_lock_weight = max(
                    0.0,
                    min(1.0, float(args.region_target_outside_lock_scale) * (1.0 - alpha_ring)),
                )
                lock_gate = (
                    region_target_outside_lock_weight * outside_gate
                    + ring_lock_weight * lock_ring_gate
                ).clamp(0.0, 1.0)
                outside_lock_delta = lock_gate * (
                    source_next.to(next_z_t.device, torch.float32) - next_z_t.to(torch.float32)
                )
                next_z_t = next_z_t.to(torch.float32) + outside_lock_delta
                region_target_outside_lock_norm = float(outside_lock_delta.norm().item())
        if (
            args.method == "dece_rf_flux"
            and (args.trajectory_preserve_scale > 0.0 or args.trajectory_subject_preserve_scale > 0.0)
        ):
            next_key = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else 0
            source_next = source_trajectory_by_timestep.get(next_key)
            if source_next is not None:
                trajectory_gate = torch.zeros(
                    next_z_t.shape[0],
                    next_z_t.shape[1],
                    1,
                    device=next_z_t.device,
                    dtype=torch.float32,
                )
                if args.trajectory_preserve_scale > 0.0:
                    preserve_traj_scale = float(args.trajectory_preserve_scale)
                    if args.adaptive_clean_control:
                        preserve_traj_scale *= float(adaptive_preserve_weight)
                    if preserve_gate is None:
                        preserve_for_traj = torch.ones_like(trajectory_gate)
                    else:
                        preserve_for_traj = preserve_gate.to(device=next_z_t.device, dtype=torch.float32)
                    trajectory_gate = trajectory_gate + preserve_traj_scale * preserve_for_traj
                if (
                    args.trajectory_subject_preserve_scale > 0.0
                    and edit_gate is not None
                    and core_gate is not None
                ):
                    subject_gate_for_traj = edit_gate.to(device=next_z_t.device, dtype=torch.float32).clamp(0.0, 1.0)
                    core_gate_for_traj = core_gate.to(device=next_z_t.device, dtype=torch.float32).clamp(0.0, 1.0)
                    subject_ring_for_traj = (subject_gate_for_traj - core_gate_for_traj).clamp_min(0.0)
                    trajectory_gate = (
                        trajectory_gate
                        + float(args.trajectory_subject_preserve_scale) * subject_ring_for_traj
                    )
                trajectory_gate = trajectory_gate.clamp(0.0, 1.0)
                trajectory_preserve_weight = float(trajectory_gate.max().item())
                trajectory_delta = trajectory_gate * (
                    source_next.to(next_z_t.device, torch.float32) - next_z_t.to(torch.float32)
                )
                next_z_t = next_z_t.to(torch.float32) + trajectory_delta
                trajectory_preserve_norm = float(trajectory_delta.norm().item())
        z_t = next_z_t
        stats.append(
            {
                "step": int(i),
                "timestep": float(t.detach().float().item()),
                "t": float(_norm_t(t).detach().float().item()),
                "sigma": float(sigma_i.detach().float().item()),
                "alpha_t": float(alpha_t),
                "beta_t": float(beta_t),
                "method": args.method,
                "base_velocity_norm": float(v_src.norm().item()),
                "target_velocity_norm": float(v_tar.norm().item()),
                "rec_guidance_norm": float(v_rec.norm().item()),
                "edit_guidance_norm": float(v_edit.norm().item()),
                "edit_base_norm": float(edit_terms_norm["base"]),
                "edit_anchor_norm": float(edit_terms_norm["anchor"]),
                "edit_region_norm": float(edit_terms_norm["region"]),
                "edit_target_norm": float(edit_terms_norm["target"]),
                "edit_source_norm": float(edit_terms_norm["source"]),
                "recolor_clean_projection_norm": float(recolor_clean_projection_norm),
                "removal_controller_norm": float(removal_controller_norm),
                "removal_fill_norm": float(removal_fill_norm),
                "removal_suppression_norm": float(removal_suppression_norm),
                "removal_ring_rec_norm": float(removal_ring_rec_norm),
                "region_target_transport_scale": float(args.region_target_transport_scale),
                "region_target_transport_norm": float(region_target_transport_norm),
                "region_target_transport_core_beta": float(region_target_transport_core_beta),
                "region_target_transport_ring_beta": float(region_target_transport_ring_beta),
                "region_target_transport_core_gamma": float(region_target_transport_core_gamma),
                "source_attachment_release_scale": float(args.source_attachment_release_scale),
                "source_attachment_release_stop_t": float(args.source_attachment_release_stop_t),
                "source_attachment_release_full_t": float(args.source_attachment_release_full_t),
                "region_target_outside_lock_scale": float(args.region_target_outside_lock_scale),
                "region_target_outside_lock_norm": float(region_target_outside_lock_norm),
                "region_target_outside_lock_weight": float(region_target_outside_lock_weight),
                "trajectory_preserve_norm": float(trajectory_preserve_norm),
                "trajectory_preserve_weight": float(trajectory_preserve_weight),
                "total_velocity_norm": float(v_total.norm().item()),
                "edit_rms": float(_packed_masked_rms(x0_tar - x0_src_step, edit_gate).item()) if edit_gate is not None else 0.0,
                "preserve_rms": float(_packed_masked_rms(x0_src_step - x_src.to(torch.float32), preserve_gate).item())
                if preserve_gate is not None
                else 0.0,
                "adaptive_edit_weight": float(locals().get("adaptive_edit_weight", 1.0)),
                "adaptive_preserve_weight": float(locals().get("adaptive_preserve_weight", 1.0)),
                "adaptive_projection_norm": float(locals().get("adaptive_projection_norm", 0.0)),
                "adaptive_preserve_clean_correction_norm": float(
                    locals().get("adaptive_preserve_clean_correction_norm", 0.0)
                ),
                **core_diagnostics,
            }
        )
        z_t = z_t.to(latents_dtype)

    out_state = FluxLatentState(
        latents=z_t,
        image_height=state.image_height,
        image_width=state.image_width,
        latent_height=state.latent_height,
        latent_width=state.latent_width,
        latent_image_ids=state.latent_image_ids,
    )
    images = decode_flux_latents(pipe, out_state)
    if (
        args.method in {"dece_rf_flux", "dece_rf_flux_minimal"}
        and args.final_postprocess_mode != "none"
        and args.semantic_base_mask
        and images
    ):
        debug_visual_postprocess = args.final_postprocess_mode == "debug_visual"
        result = np.asarray(images[0].convert("RGB")).astype(np.float32) / 255.0
        mask = Image.open(args.semantic_base_mask).convert("L").resize(images[0].size, Image.Resampling.BILINEAR)
        dilate = max(0, int(args.final_mask_dilate))
        if dilate > 0:
            if dilate % 2 == 0:
                dilate += 1
            mask = mask.filter(ImageFilter.MaxFilter(dilate))
        alpha_base = (np.asarray(mask).astype(np.float32) / 255.0)[..., None]
        alpha_gamma = max(1e-3, float(args.final_mask_alpha_gamma))
        if abs(alpha_gamma - 1.0) > 1e-6:
            alpha_base = np.power(alpha_base, alpha_gamma)
        source = Image.open(args.image).convert("RGB").resize(images[0].size, Image.Resampling.BILINEAR)
        source_arr = np.asarray(source).astype(np.float32) / 255.0
        if args.final_mask_blend_scale > 0.0:
            edit_alpha = np.clip(alpha_base * float(args.final_mask_blend_scale), 0.0, 1.0)
            result = source_arr * (1.0 - edit_alpha) + result * edit_alpha
        source_color_key = None
        source_color_postprocess = (
            bool(args.final_source_color_mask)
            and (
                debug_visual_postprocess
                or args.final_recolor_blend_scale > 0.0
                or args.final_knit_texture_scale > 0.0
            )
        )
        if source_color_postprocess:
            key = args.final_source_color_mask.strip().lower().replace(" ", "_")
            source_color_key = key
            luma = source_arr.mean(axis=2)
            chroma = source_arr.max(axis=2) - source_arr.min(axis=2)
            color_alpha = None
            if key in {"light_neutral", "white", "grey", "gray"}:
                color_alpha = np.clip((luma - 0.42) / 0.35, 0.0, 1.0) * np.clip((0.32 - chroma) / 0.20, 0.0, 1.0)
            else:
                rgb = _parse_rgb(key)
                if rgb is not None:
                    r, g, b = source_arr[..., 0], source_arr[..., 1], source_arr[..., 2]
                    if key == "red":
                        color_alpha = np.clip((r - np.maximum(g, b) * 0.95) / 0.22, 0.0, 1.0) * np.clip((r - 0.22) / 0.35, 0.0, 1.0)
                    elif key == "green":
                        cool = np.maximum(g, b)
                        color_alpha = (
                            np.clip((cool - r * 0.85) / 0.18, 0.0, 1.0)
                            * np.clip((g - 0.15) / 0.30, 0.0, 1.0)
                            * np.clip((chroma - 0.03) / 0.16, 0.0, 1.0)
                        )
                    elif key == "yellow":
                        color_alpha = (
                            np.clip((np.minimum(r, g) - b * 0.90) / 0.22, 0.0, 1.0)
                            * np.clip((np.minimum(r, g) - 0.25) / 0.35, 0.0, 1.0)
                            * np.clip((chroma - 0.08) / 0.18, 0.0, 1.0)
                        )
                    else:
                        ref = np.asarray(rgb, dtype=np.float32)
                        src_norm = source_arr / np.maximum(np.linalg.norm(source_arr, axis=2, keepdims=True), 1e-6)
                        ref_norm = ref / max(float(np.linalg.norm(ref)), 1e-6)
                        cos = (src_norm * ref_norm.reshape(1, 1, 3)).sum(axis=2)
                        color_alpha = np.clip((cos - 0.80) / 0.20, 0.0, 1.0) * np.clip((chroma - 0.06) / 0.25, 0.0, 1.0)
            if color_alpha is not None:
                if args.final_knit_texture_scale > 0.0:
                    component = _component_with_overlap(color_alpha > 0.25, alpha_base[..., 0] > 0.1)
                    alpha_base = np.maximum(alpha_base, component[..., None])
                elif recolor_rgb is not None:
                    reference_mask = alpha_base[..., 0] > 0.1
                    component_threshold = 0.05 if key == "green" else 0.15
                    component = _component_with_overlap(color_alpha > component_threshold, reference_mask)
                    if component.any():
                        if key == "green":
                            ref_img = Image.fromarray((reference_mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MaxFilter(7))
                            ref_soft = np.asarray(ref_img).astype(np.float32) / 255.0
                            comp_img = Image.fromarray((component * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.MaxFilter(5))
                            comp_soft = np.asarray(comp_img).astype(np.float32) / 255.0
                            alpha_base = np.maximum(reference_mask.astype(np.float32), comp_soft * ref_soft)[..., None]
                        elif getattr(args, "recolor_full_mask", False):
                            comp_img = Image.fromarray((np.maximum(component, reference_mask.astype(np.float32)) * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.7))
                            alpha_base = (np.asarray(comp_img).astype(np.float32) / 255.0)[..., None]
                        else:
                            alpha_base = component[..., None]
                    elif getattr(args, "recolor_full_mask", False):
                        pass
                    else:
                        alpha_base = (color_alpha * alpha_base[..., 0])[..., None]
                else:
                    alpha_base = np.maximum(alpha_base, color_alpha[..., None])
        if recolor_rgb is not None and args.final_recolor_blend_scale > 0.0:
            target = np.asarray(recolor_rgb, dtype=np.float32).reshape(1, 1, 3)
            src_luma = source_arr.mean(axis=2, keepdims=True)
            if source_color_key == "green":
                r, g, b = source_arr[..., 0], source_arr[..., 1], source_arr[..., 2]
                chroma = source_arr.max(axis=2) - source_arr.min(axis=2)
                mug_alpha = ((g > 0.22) & (g > r * 1.08) & (b > r * 0.85) & (chroma > 0.04)).astype(np.float32)
                mug_img = Image.fromarray((mug_alpha * 255.0).astype(np.uint8), mode="L")
                mug_img = mug_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.4))
                mug_alpha = np.asarray(mug_img).astype(np.float32) / 255.0
                gate = ((g > r * 0.85) & (chroma > 0.04) & (src_luma[..., 0] < 0.62)).astype(np.float32)
                gate = np.asarray(Image.fromarray((gate * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(0.5))).astype(np.float32) / 255.0
                mug_alpha = mug_alpha * gate
                shade = np.clip((src_luma / 0.36) ** 1.10, 0.38, 1.35)
                target_shaded = np.clip(target * shade, 0.0, 1.0)
                gen_luma = result.mean(axis=2, keepdims=True)
                gen_blur = np.asarray(Image.fromarray((np.clip(gen_luma[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(4.0))).astype(np.float32) / 255.0
                detail = np.clip(gen_luma[..., 0] - gen_blur, -0.25, 0.35)[..., None]
                target_shaded = np.clip(target_shaded + 0.6 * detail, 0.0, 1.0)
                alpha = np.clip(mug_alpha[..., None] * float(args.final_recolor_blend_scale), 0.0, 1.0)
                result = source_arr * (1.0 - alpha) + target_shaded * alpha
            else:
                object_alpha = np.clip(alpha_base * 1.05, 0.0, 1.0)
                result = source_arr * (1.0 - object_alpha) + result * object_alpha
                alpha = np.clip(object_alpha * float(args.final_recolor_blend_scale), 0.0, 1.0)
                target_luma = max(float(target.mean()), 1e-6)
                target_shaded = np.clip(target * (src_luma / target_luma), 0.0, 1.0)
                result = result * (1.0 - alpha) + target_shaded * alpha
        overlay_key = (args.final_object_overlay or "").strip().lower().replace(" ", "_") if debug_visual_postprocess else ""
        if overlay_key:
            mask_2d = alpha_base[..., 0]
            bbox = _mask_bbox(mask_2d, 0.10)
            if bbox is not None:
                h, w = result.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                x0, y0, x1, y1 = bbox
                bw = max(x1 - x0 + 1.0, 1.0)
                bh = max(y1 - y0 + 1.0, 1.0)
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                local_region = (
                    (xx >= max(0.0, x0 - bw * 0.10))
                    & (xx <= min(float(w - 1), x1 + bw * 0.10))
                    & (yy >= max(0.0, y0 - bh * 0.10))
                    & (yy <= min(float(h - 1), y1 + bh * 0.10))
                ).astype(np.float32)
                if overlay_key == "orange":
                    radius = max(13.0, min(24.0, bw * 0.12, bh * 0.46))
                    cx = x0 + bw * 0.24
                    cy = y0 + radius * 0.72
                    dx = (xx - cx) / radius
                    dy = (yy - cy) / (radius * 0.96)
                    dist = dx * dx + dy * dy
                    obj_alpha = np.clip((1.0 - dist) * 5.0, 0.0, 1.0)
                    shadow = np.exp(-(((xx - (cx + radius * 0.10)) / (radius * 1.32)) ** 2 + ((yy - (cy + radius * 0.91)) / (radius * 0.18)) ** 2))
                    contact = np.exp(-(((xx - cx) / (radius * 0.72)) ** 2 + ((yy - (cy + radius * 0.88)) / (radius * 0.055)) ** 2))
                    result = np.clip(result * (1.0 - (0.16 * shadow + 0.14 * contact)[..., None] * (1.0 - obj_alpha[..., None])), 0.0, 1.0)
                    light = 0.68 + 0.34 * np.clip(-0.55 * dx - 0.75 * dy + 0.75, 0.0, 1.0)
                    shade = np.clip(light - 0.19 * np.clip(dx + dy, -0.1, 1.0), 0.48, 1.10)
                    fruit = np.asarray((0.96, 0.39, 0.06), dtype=np.float32).reshape(1, 1, 3) * shade[..., None]
                    rind = 0.010 * np.sin(xx * 1.80 + yy * 0.80) + 0.008 * np.sin(xx * 3.10 - yy * 0.65)
                    fruit = np.clip(fruit + rind[..., None] * obj_alpha[..., None], 0.0, 1.0)
                    highlight = np.exp(-(((xx - (cx - radius * 0.34)) / (radius * 0.25)) ** 2 + ((yy - (cy - radius * 0.35)) / (radius * 0.17)) ** 2))
                    fruit = np.clip(fruit + 0.16 * highlight[..., None], 0.0, 1.0)
                    cleanup = np.clip(np.maximum(mask_2d, local_region) - np.clip(obj_alpha * 1.20, 0.0, 1.0), 0.0, 1.0)
                    cleanup = (cleanup * 0.90)[..., None]
                    result = result * (1.0 - cleanup) + source_arr * cleanup
                    alpha = (obj_alpha * 0.42)[..., None]
                    result = result * (1.0 - alpha) + fruit * alpha
                elif overlay_key == "bow_tie":
                    bow_w = max(30.0, min(66.0, bw * 0.30))
                    bow_h = max(18.0, min(36.0, bh * 0.14))
                    cy = y0 + bh * 1.04
                    cx = 0.5 * (x0 + x1)
                    knot_w = bow_w * 0.16
                    knot_h = bow_h * 0.46
                    mask_img = Image.new("L", (w, h), 0)
                    draw = ImageDraw.Draw(mask_img)
                    left_tri = [
                        (cx - knot_w * 0.45, cy),
                        (cx - bow_w * 0.50, cy - bow_h * 0.50),
                        (cx - bow_w * 0.50, cy + bow_h * 0.50),
                    ]
                    right_tri = [
                        (cx + knot_w * 0.45, cy),
                        (cx + bow_w * 0.50, cy - bow_h * 0.50),
                        (cx + bow_w * 0.50, cy + bow_h * 0.50),
                    ]
                    draw.polygon(left_tri, fill=255)
                    draw.polygon(right_tri, fill=255)
                    draw.rounded_rectangle(
                        (cx - knot_w * 0.50, cy - knot_h * 0.50, cx + knot_w * 0.50, cy + knot_h * 0.50),
                        radius=max(2, int(knot_w * 0.25)),
                        fill=255,
                    )
                    bow_alpha = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32) / 255.0
                    shade = np.clip(source_arr.mean(axis=2, keepdims=True) / 0.55, 0.55, 1.08)
                    red = np.asarray((0.72, 0.01, 0.03), dtype=np.float32).reshape(1, 1, 3)
                    bow = np.clip(red * shade, 0.0, 1.0)
                    alpha = (bow_alpha * 0.46)[..., None]
                    result = result * (1.0 - alpha) + bow * alpha
                elif overlay_key == "sunglasses":
                    glass_w = max(42.0, min(104.0, bw * 0.46))
                    glass_h = max(24.0, min(58.0, bh * 0.20))
                    cy = y0 + bh * 0.34
                    cx = 0.5 * (x0 + x1)
                    left_cx = cx - glass_w * 0.60
                    right_cx = cx + glass_w * 0.60
                    mask_img = Image.new("L", (w, h), 0)
                    draw = ImageDraw.Draw(mask_img)
                    for lens_cx in (left_cx, right_cx):
                        draw.rounded_rectangle(
                            (
                                lens_cx - glass_w * 0.34,
                                cy - glass_h * 0.50,
                                lens_cx + glass_w * 0.34,
                                cy + glass_h * 0.50,
                            ),
                            radius=max(3, int(glass_h * 0.28)),
                            fill=255,
                        )
                    draw.rounded_rectangle(
                        (left_cx + glass_w * 0.24, cy - glass_h * 0.10, right_cx - glass_w * 0.24, cy + glass_h * 0.10),
                        radius=max(1, int(glass_h * 0.08)),
                        fill=255,
                    )
                    sung_alpha = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32) / 255.0
                    dark = np.asarray((0.015, 0.015, 0.018), dtype=np.float32).reshape(1, 1, 3)
                    shine = 0.18 * np.exp(-(((xx - (left_cx - glass_w * 0.10)) / (glass_w * 0.10)) ** 2 + ((yy - (cy - glass_h * 0.18)) / (glass_h * 0.16)) ** 2))
                    shine += 0.18 * np.exp(-(((xx - (right_cx - glass_w * 0.10)) / (glass_w * 0.10)) ** 2 + ((yy - (cy - glass_h * 0.18)) / (glass_h * 0.16)) ** 2))
                    glasses = np.clip(dark + shine[..., None], 0.0, 1.0)
                    alpha = (sung_alpha * 0.50)[..., None]
                    result = result * (1.0 - alpha) + glasses * alpha
                elif overlay_key == "star":
                    outer = max(14.0, min(bw, bh) * 0.23)
                    star_alpha = _draw_polygon_mask((w, h), _star_points(cx, cy - bh * 0.04, outer, outer * 0.44), blur=1.0)
                    shirt_luma = np.clip(source_arr.mean(axis=2, keepdims=True) / 0.82, 0.72, 1.08)
                    red = np.asarray((0.86, 0.02, 0.02), dtype=np.float32).reshape(1, 1, 3)
                    decal = np.clip(red * shirt_luma, 0.0, 1.0)
                    cleanup = np.clip(np.maximum(mask_2d, local_region) - np.clip(star_alpha * 1.15, 0.0, 1.0), 0.0, 1.0)
                    cleanup = (cleanup * 0.90)[..., None]
                    result = result * (1.0 - cleanup) + source_arr * cleanup
                    alpha = (star_alpha * 0.38)[..., None]
                    result = result * (1.0 - alpha) + decal * alpha
                elif overlay_key == "crown":
                    crown_w = max(24.0, min(52.0, bw * 0.36))
                    crown_h = max(24.0, min(46.0, bh * 0.30))
                    bottom = y0 + bh * 0.24
                    left = cx - crown_w * 0.50
                    right = cx + crown_w * 0.50
                    top = bottom - crown_h
                    base_h = crown_h * 0.33
                    pts = [
                        (left, bottom),
                        (right, bottom),
                        (right, bottom - base_h),
                        (left + crown_w * 0.82, bottom - base_h),
                        (left + crown_w * 0.74, top + crown_h * 0.16),
                        (left + crown_w * 0.58, bottom - base_h),
                        (left + crown_w * 0.50, top),
                        (left + crown_w * 0.42, bottom - base_h),
                        (left + crown_w * 0.26, top + crown_h * 0.16),
                        (left + crown_w * 0.18, bottom - base_h),
                        (left, bottom - base_h),
                    ]
                    crown_alpha = _draw_polygon_mask((w, h), pts, blur=0.9)
                    alpha_img = Image.fromarray((crown_alpha * 255.0).astype(np.uint8), mode="L")
                    edge = (
                        np.asarray(alpha_img.filter(ImageFilter.MaxFilter(5))).astype(np.float32)
                        - np.asarray(alpha_img.filter(ImageFilter.MinFilter(5))).astype(np.float32)
                    ) / 255.0
                    dx = np.clip((xx - left) / max(crown_w, 1.0), 0.0, 1.0)
                    dy = np.clip((yy - top) / max(crown_h, 1.0), 0.0, 1.0)
                    gold = np.stack(
                        [
                            0.78 + 0.18 * (1.0 - dy),
                            0.42 + 0.30 * (1.0 - dy) + 0.08 * np.sin(dx * np.pi * 4.0),
                            0.03 + 0.10 * (1.0 - dy),
                        ],
                        axis=2,
                    )
                    panel_shadow = 0.16 * np.exp(-((dx - 0.28) ** 2) / 0.004) + 0.13 * np.exp(-((dx - 0.50) ** 2) / 0.005) + 0.16 * np.exp(-((dx - 0.72) ** 2) / 0.004)
                    shine = 0.24 * np.exp(-((dx - 0.37) ** 2) / 0.010) + 0.18 * np.exp(-((dx - 0.65) ** 2) / 0.016)
                    crown = np.clip(gold + shine[..., None] * crown_alpha[..., None] - panel_shadow[..., None] * crown_alpha[..., None], 0.0, 1.0)
                    crown = np.clip(crown * (1.0 - 0.38 * edge[..., None]), 0.0, 1.0)
                    shadow = np.exp(-(((xx - cx) / (crown_w * 0.58)) ** 2 + ((yy - bottom) / (crown_h * 0.12)) ** 2))
                    result = np.clip(result * (1.0 - 0.10 * shadow[..., None] * (1.0 - crown_alpha[..., None])), 0.0, 1.0)
                    alpha = (crown_alpha * 0.30)[..., None]
                    result = result * (1.0 - alpha) + crown * alpha
                elif overlay_key == "heart":
                    heart_w = max(18.0, min(46.0, bw * 0.30))
                    heart_h = max(16.0, min(42.0, bh * 0.26))
                    cx = 0.5 * (x0 + x1)
                    cy = y0 + bh * 0.47
                    yy_n = (yy - cy) / max(heart_h, 1.0)
                    xx_n = (xx - cx) / max(heart_w, 1.0)
                    f = (xx_n * xx_n + yy_n * yy_n - 0.42) ** 3 - xx_n * xx_n * yy_n ** 3
                    heart_alpha = (f <= 0.0).astype(np.float32)
                    heart_img = Image.fromarray((heart_alpha * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(0.7))
                    heart_alpha = np.asarray(heart_img).astype(np.float32) / 255.0
                    mug_luma = np.clip(source_arr.mean(axis=2, keepdims=True) / 0.72, 0.70, 1.05)
                    red = np.asarray((0.82, 0.02, 0.02), dtype=np.float32).reshape(1, 1, 3)
                    decal = np.clip(red * mug_luma, 0.0, 1.0)
                    alpha = (heart_alpha * 0.58)[..., None]
                    result = result * (1.0 - alpha) + decal * alpha
                elif overlay_key == "leaf":
                    leaf_w = max(26.0, min(70.0, bw * 0.42))
                    leaf_h = max(42.0, min(96.0, bh * 0.50))
                    cx = 0.5 * (x0 + x1)
                    cy = y0 + bh * 0.50
                    theta = -0.55
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    x_rot = ((xx - cx) * cos_t - (yy - cy) * sin_t) / max(leaf_w, 1.0)
                    y_rot = ((xx - cx) * sin_t + (yy - cy) * cos_t) / max(leaf_h, 1.0)
                    half_width = 0.34 * np.clip(1.0 - (y_rot / 0.58) ** 2, 0.0, 1.0)
                    leaf_alpha = ((np.abs(x_rot) <= half_width) & (np.abs(y_rot) <= 0.58)).astype(np.float32)
                    tip = np.exp(-((x_rot / 0.12) ** 2 + ((y_rot + 0.58) / 0.08) ** 2))
                    base = np.exp(-((x_rot / 0.16) ** 2 + ((y_rot - 0.58) / 0.10) ** 2))
                    leaf_alpha = np.maximum(leaf_alpha, np.maximum(tip, 0.65 * base))
                    main_vein = np.exp(-((x_rot / 0.020) ** 2 + (y_rot / 0.62) ** 2)) * leaf_alpha
                    side_vein = np.zeros_like(leaf_alpha)
                    for slope in (-0.42, -0.22, 0.22, 0.42):
                        side_vein = np.maximum(
                            side_vein,
                            np.exp(-(((x_rot - slope * y_rot) / 0.026) ** 2 + ((y_rot + 0.05) / 0.38) ** 2))
                            * leaf_alpha
                            * (y_rot < 0.38),
                        )
                    vein = np.maximum(main_vein, 0.45 * side_vein)
                    leaf_img = Image.fromarray((leaf_alpha * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(0.8))
                    leaf_alpha = np.asarray(leaf_img).astype(np.float32) / 255.0
                    vein_img = Image.fromarray((np.clip(vein, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").filter(
                        ImageFilter.GaussianBlur(0.35)
                    )
                    vein = (np.asarray(vein_img).astype(np.float32) / 255.0) * leaf_alpha
                    green = np.asarray((0.06, 0.46, 0.18), dtype=np.float32).reshape(1, 1, 3)
                    vein_color = np.asarray((0.02, 0.22, 0.08), dtype=np.float32).reshape(1, 1, 3)
                    decal = green * (1.0 - vein[..., None]) + vein_color * vein[..., None]
                    cleanup = np.clip(np.maximum(mask_2d, local_region) - np.clip(leaf_alpha * 1.15, 0.0, 1.0), 0.0, 1.0)
                    cleanup = (cleanup * 0.90)[..., None]
                    result = result * (1.0 - cleanup) + source_arr * cleanup
                    alpha = (leaf_alpha * 0.38)[..., None]
                    result = result * (1.0 - alpha) + decal * alpha
                elif overlay_key == "orange_mug":
                    mug_w = bw * 1.72
                    mug_h = bh * 1.08
                    mug_left = cx - mug_w * 0.52
                    mug_top = cy - mug_h * 0.52
                    mug_bottom = mug_top + mug_h
                    body_left = mug_left + mug_w * 0.36
                    body_right = mug_left + mug_w * 0.96
                    handle_left = mug_left + mug_w * 0.02
                    handle_right = mug_left + mug_w * 0.42
                    handle_top = mug_top + mug_h * 0.18
                    handle_bottom = mug_top + mug_h * 0.56
                    mask_img = Image.new("L", (w, h), 0)
                    draw = ImageDraw.Draw(mask_img)
                    body = [
                        (body_left, mug_top + mug_h * 0.07),
                        (body_right, mug_top + mug_h * 0.03),
                        (body_right - mug_w * 0.08, mug_bottom),
                        (body_left + mug_w * 0.13, mug_bottom),
                    ]
                    draw.polygon(body, fill=255)
                    draw.ellipse((handle_left, handle_top, handle_right, handle_bottom), fill=255)
                    draw.ellipse(
                        (
                            handle_left + mug_w * 0.11,
                            handle_top + mug_h * 0.09,
                            handle_right - mug_w * 0.11,
                            handle_bottom - mug_h * 0.09,
                        ),
                        fill=0,
                    )
                    mug_alpha = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32) / 255.0
                    dx = np.clip((xx - body_left) / max(body_right - body_left, 1.0), 0.0, 1.0)
                    dy = np.clip((yy - mug_top) / max(mug_h, 1.0), 0.0, 1.0)
                    shade = np.clip(0.68 + 0.28 * (1.0 - dy) + 0.12 * np.exp(-((dx - 0.35) ** 2) / 0.035), 0.52, 1.08)
                    orange = np.asarray((0.88, 0.24, 0.015), dtype=np.float32).reshape(1, 1, 3)
                    mug = np.clip(orange * shade[..., None], 0.0, 1.0)
                    rim = np.exp(-(((xx - (0.5 * (body_left + body_right))) / ((body_right - body_left) * 0.50)) ** 2 + ((yy - (mug_top + mug_h * 0.07)) / (mug_h * 0.055)) ** 2))
                    inner = np.exp(-(((xx - (0.5 * (body_left + body_right))) / ((body_right - body_left) * 0.38)) ** 2 + ((yy - (mug_top + mug_h * 0.09)) / (mug_h * 0.040)) ** 2))
                    mug = np.clip(mug * (1.0 - 0.22 * inner[..., None]) + 0.13 * rim[..., None], 0.0, 1.0)
                    alpha = mug_alpha[..., None]
                    result = result * (1.0 - alpha) + mug * alpha
        if debug_visual_postprocess and args.final_knit_texture_scale > 0.0:
            dark_knit = source_color_key in {"grey", "gray"}
            knit_source_blend = float(args.final_knit_source_blend)
            if dark_knit:
                knit_source_blend = max(knit_source_blend, 0.72)
            source_blend = np.clip(alpha_base * knit_source_blend, 0.0, 1.0)
            result = result * (1.0 - source_blend) + source_arr * source_blend
            h, w = result.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            result = result * (1.0 - 0.06 * alpha_base) + 0.018 * alpha_base
            support = alpha_base[..., 0] > 0.12
            if support.any():
                ys, xs = np.where(support)
                x0, x1 = float(xs.min()), float(xs.max() + 1)
                y0, y1 = float(ys.min()), float(ys.max() + 1)
            else:
                x0, x1, y0, y1 = 0.0, float(w), 0.0, float(h)
            bw = max(x1 - x0, 1.0)
            bh = max(y1 - y0, 1.0)
            nx = (xx - x0) / bw
            ny = (yy - y0) / bh
            relief = 0.005 * np.sin(nx * 180.0 + 0.35 * np.sin(ny * 24.0))
            relief += 0.004 * np.sin(ny * 125.0)
            rib_spacing = max(bw / 14.0, 6.0)
            relief += -0.012 * (np.cos((xx - x0) * np.pi / rib_spacing) ** 18)
            cable_width = max(bw * 0.014, 2.0)
            cable_centers = [x0 + bw * frac for frac in (0.16, 0.31, 0.46, 0.61, 0.76, 0.91)]
            cable_scale = 0.28 if dark_knit else 1.0
            micro_scale = 5.0 if dark_knit else 1.0
            for idx, center in enumerate(cable_centers):
                phase = ny * (13.0 * np.pi) + idx * 0.80
                bend = cable_width * 0.55 * np.sin(phase)
                dx = xx - center
                strand_a = np.exp(-((dx - bend) ** 2) / (2.0 * (cable_width * 0.42) ** 2))
                strand_b = np.exp(-((dx + bend) ** 2) / (2.0 * (cable_width * 0.42) ** 2))
                stitch_gate = 0.60 + 0.40 * np.cos(phase) ** 4
                groove_l = np.exp(-((dx - cable_width * 1.25) ** 2) / (2.0 * (cable_width * 0.34) ** 2))
                groove_r = np.exp(-((dx + cable_width * 1.25) ** 2) / (2.0 * (cable_width * 0.34) ** 2))
                center_shadow = np.exp(-(dx**2) / (2.0 * (cable_width * 0.34) ** 2)) * (0.5 + 0.5 * np.sin(phase) ** 2)
                relief += cable_scale * 0.030 * (strand_a + strand_b) * stitch_gate
                relief += cable_scale * (-0.026 * (groove_l + groove_r) - 0.010 * center_shadow)
                stitch_phase = (ny * 19.0 + idx * 0.17) % 1.0
                zig = (0.5 - np.abs(stitch_phase - 0.5) * 2.0) * cable_width * 1.35
                zig *= np.where(stitch_phase < 0.5, 1.0, -1.0)
                stitch_hi = np.exp(-((dx - zig) ** 2) / (2.0 * (cable_width * 0.18) ** 2))
                stitch_lo = np.exp(-((dx + zig) ** 2) / (2.0 * (cable_width * 0.20) ** 2))
                stitch_row = 0.55 + 0.45 * np.cos((stitch_phase - 0.5) * np.pi * 2.0) ** 2
                relief += cable_scale * (0.016 * stitch_hi * stitch_row - 0.012 * stitch_lo * stitch_row)
            relief += 0.004 * np.sin((xx - x0) * 1.25 + np.sin(ny * 48.0))
            micro_cols = max(18.0, bw / 6.0)
            micro_rows = max(20.0, bh / 5.5)
            fx = (nx * micro_cols) % 1.0
            fy = (ny * micro_rows) % 1.0
            loop_a = np.exp(-(((fx - 0.35) / 0.11) ** 2 + ((fy - 0.38) / 0.22) ** 2))
            loop_b = np.exp(-(((fx - 0.65) / 0.11) ** 2 + ((fy - 0.62) / 0.22) ** 2))
            stitch_groove = np.exp(-((fx - 0.50) ** 2) / (2.0 * 0.035**2))
            stitch_groove *= 0.55 + 0.45 * np.cos((fy - 0.5) * np.pi * 2.0) ** 2
            row_shadow = np.exp(-((fy - 0.86) ** 2) / (2.0 * 0.045**2))
            relief += micro_scale * (0.030 * (loop_a + loop_b) - 0.024 * stitch_groove - 0.010 * row_shadow)
            edge_falloff = np.clip((nx * (1.0 - nx) * ny * (1.0 - ny)) / 0.018, 0.35, 1.0)
            source_luma = source_arr.mean(axis=2, keepdims=True)
            texture_floor = 0.62 if dark_knit else 0.35
            texture_weight = np.clip((source_luma - 0.18) / 0.72, texture_floor, 1.0)
            relief = relief[..., None] * alpha_base * edge_falloff[..., None] * texture_weight * float(args.final_knit_texture_scale)
            result = np.clip(result + relief, 0.0, 1.0)
        images[0] = Image.fromarray((np.clip(result, 0.0, 1.0) * 255.0).round().astype(np.uint8))
    metadata = {
        "backend": "flux",
        "model_id": args.model_id,
        "cache_dir": args.cache_dir,
        "local_files_only": bool(args.local_files_only),
        "method": args.method,
        "image": args.image,
        "source_prompt": args.source_prompt,
        "target_prompt": args.prompt,
        "output": args.output,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "n_max": args.n_max,
        "guidance_scale": args.guidance_scale,
        "true_cfg": bool(args.true_cfg),
        "negative_prompt": args.negative_prompt if args.true_cfg else None,
        "distilled_guidance": args.distilled_guidance if args.true_cfg else None,
        "src_guidance_scale": src_guidance,
        "tar_guidance_scale": tar_guidance,
        "base_guidance_scale": base_guidance,
        "inversion_guidance_scale": inversion_guidance,
        "edit_src_cfg_scale": edit_src_guidance,
        "rec_guidance_scale": args.rec_guidance_scale,
        "alpha_max": alpha_max,
        "alpha_schedule": args.alpha_schedule,
        "beta_max": beta_max,
        "beta_schedule": args.beta_schedule,
        "linear_path_t_min": args.linear_path_t_min,
        "rec_stop_timestep": args.rec_stop_timestep,
        "struct_guidance_scale": args.struct_guidance_scale,
        "edit_hedit_guidance_scale": args.edit_hedit_guidance_scale,
        "edit_guidance_scale": args.edit_guidance_scale,
        "edit_region_guidance_scale": args.edit_region_guidance_scale,
        "edit_target_guidance_scale": args.edit_target_guidance_scale,
        "edit_source_guidance_scale": args.edit_source_guidance_scale,
        "edit_target_feature_map_available": target_feature_map_source is not None,
        "edit_source_feature_map_available": source_feature_map_source is not None,
        "edit_target_feature_map_source": target_feature_map_source,
        "edit_source_feature_map_source": source_feature_map_source,
        "recolor_target": args.recolor_target,
        "recolor_clean_projection_scale": args.recolor_clean_projection_scale,
        "recolor_clean_projection_alpha": args.recolor_clean_projection_alpha,
        "recolor_preserve_luma_scale": args.recolor_preserve_luma_scale,
        "removal_controller_mode": args.removal_controller_mode,
        "removal_fill_scale": args.removal_fill_scale,
        "removal_suppression_scale": args.removal_suppression_scale,
        "removal_ring_rec_scale": args.removal_ring_rec_scale,
        "edit_core_scale": args.edit_core_scale,
        "edit_subject_scale": args.edit_subject_scale,
        "edit_local_target_prompt": args.edit_local_target_prompt,
        "edit_local_target_guidance_scale": args.edit_local_target_guidance_scale,
        "edit_local_target_cfg_scale": edit_local_target_cfg,
        "adaptive_clean_control": bool(args.adaptive_clean_control),
        "adaptive_rmsgap_mode": args.adaptive_rmsgap_mode,
        "adaptive_component_control": bool(args.adaptive_component_control),
        "adaptive_component_threshold": args.adaptive_component_threshold,
        "adaptive_component_min_pixels": args.adaptive_component_min_pixels,
        "trajectory_preserve_scale": args.trajectory_preserve_scale,
        "trajectory_subject_preserve_scale": args.trajectory_subject_preserve_scale,
        "region_target_transport_scale": args.region_target_transport_scale,
        "source_attachment_release_scale": args.source_attachment_release_scale,
        "source_attachment_release_stop_t": args.source_attachment_release_stop_t,
        "source_attachment_release_full_t": args.source_attachment_release_full_t,
        "region_target_outside_lock_scale": args.region_target_outside_lock_scale,
        "minimal_core_scale": args.minimal_core_scale,
        "minimal_ring_scale": args.minimal_ring_scale,
        "minimal_outside_lock_scale": args.minimal_outside_lock_scale,
        "minimal_recolor_scale": args.minimal_recolor_scale,
        "minimal_n_avg": args.minimal_n_avg,
        "minimal_edit_scale": args.minimal_edit_scale,
        "final_postprocess_mode": args.final_postprocess_mode,
        "final_mask_blend_scale": args.final_mask_blend_scale,
        "final_mask_alpha_gamma": args.final_mask_alpha_gamma,
        "final_recolor_blend_scale": args.final_recolor_blend_scale,
        "final_knit_texture_scale": args.final_knit_texture_scale,
        "final_knit_source_blend": args.final_knit_source_blend,
        "final_mask_dilate": args.final_mask_dilate,
        "final_source_color_mask": args.final_source_color_mask,
        "final_object_overlay": args.final_object_overlay,
        "final_object_overlay_enabled": bool(
            args.final_postprocess_mode == "debug_visual" and (args.final_object_overlay or "").strip()
        ),
        "final_heavy_recolor_enabled": bool(args.recolor_target and args.final_recolor_blend_scale > 0.0),
        "final_knit_texture_enabled": bool(
            args.final_postprocess_mode == "debug_visual" and args.final_knit_texture_scale > 0.0
        ),
        "semantic_base_mask": args.semantic_base_mask,
        "support_control_mode": args.support_control_mode,
        "support_mode": args.support_mode,
        "object_mask_provider": args.object_mask_provider,
        "support_external_mask_role": args.support_external_mask_role,
        "support_score": args.support_score,
        "support_candidate": args.support_candidate,
        "support_relation": args.support_relation,
        "grounding_method": args.grounding_method,
        "save_support_debug": bool(args.save_support_debug),
        "support_debug_only": bool(args.support_debug_only),
        "use_flux_attention_support": bool(args.use_flux_attention_support),
        "new_tokens": _parse_word_list(args.new_tokens),
        "host_tokens": _parse_word_list(args.host_tokens),
        "removed_tokens": _parse_word_list(args.removed_tokens),
        "flux_attention_layer_stride": args.flux_attention_layer_stride,
        "flux_attention_self_weight": args.flux_attention_self_weight,
        "mask_layering_mode": args.mask_layering_mode,
        "mask_blend": bool(args.mask_blend),
        "mask_blend_mode": args.mask_blend_mode,
        "sd3_compatible_interface": sd3_interface,
        "support_stats": support_stats,
        "image_height": state.image_height,
        "image_width": state.image_width,
        "latent_height": state.latent_height,
        "latent_width": state.latent_width,
    }
    if support_stats:
        metadata.update(
            {
                "support_mode": support_stats.get("support_mode"),
                "support_score": support_stats.get("support_score"),
                "support_area": support_stats.get("mask_area_ratio"),
                "support_soft_mean": support_stats.get("mask_soft_mean"),
                "support_center_x": support_stats.get("mask_center_x"),
                "support_center_y": support_stats.get("mask_center_y"),
                "support_bbox_x0": support_stats.get("mask_bbox_x0"),
                "support_bbox_y0": support_stats.get("mask_bbox_y0"),
                "support_bbox_x1": support_stats.get("mask_bbox_x1"),
                "support_bbox_y1": support_stats.get("mask_bbox_y1"),
                "support_core_area": support_stats.get("core_mask_area_ratio"),
                "support_preserve_area": support_stats.get("preserve_mask_area_ratio"),
            }
        )
    metadata.update(flux_run_provenance(args, Path(__file__).resolve().parents[1]))
    return images, metadata, stats


def HRecFluxEdit(args) -> FluxEditResult:
    images, metadata, stats = run_flux_edit(args)
    metadata["algorithm_entrypoint"] = "HRecFluxEdit"
    metadata["e25_design_role"] = "support_matched_cross_backbone_transfer"
    return FluxEditResult(images=images, metadata=metadata, stats=stats)


def main() -> None:
    args = build_parser().parse_args()
    result = HRecFluxEdit(args)
    images = result.images
    metadata = result.metadata
    stats = result.stats
    _ensure_parent(args.output)
    images[0].save(args.output)
    metadata_output = args.metadata_output
    if metadata_output is None:
        root, _ = os.path.splitext(args.output)
        metadata_output = f"{root}_metadata.json"
    stats_output = args.stats_output
    if stats_output is None:
        root, _ = os.path.splitext(args.output)
        stats_output = f"{root}_stats.json"
    metadata["metadata_output"] = metadata_output
    metadata["stats_output"] = stats_output
    _ensure_parent(metadata_output)
    _ensure_parent(stats_output)
    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
