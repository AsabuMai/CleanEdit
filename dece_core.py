from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from energies import (
    clean_delta_to_velocity,
    editing_velocity_surrogate_total,
    reconstruction_velocity_surrogate_total,
)


TensorMapFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class DeceCoreConfig:
    """Backend-neutral DeCE-RF velocity-control weights for one RF step."""

    struct_guidance_scale: float = 0.0
    edit_hedit_guidance_scale: float = 1.0
    edit_guidance_scale: float = 0.0
    edit_region_guidance_scale: float = 0.0
    edit_target_guidance_scale: float = 0.0
    edit_source_guidance_scale: float = 0.0
    linear_path_t_min: float = 1e-3

    adaptive_clean_control: bool = False
    adaptive_rmsgap_mode: str = "normgate"
    adaptive_rmsgap_dead_zone: float = 0.0
    adaptive_rmsgap_preserve_gate_budget: float = 0.0
    adaptive_edit_target_progress: float = 0.0
    adaptive_edit_target_rms: float = 0.0
    adaptive_preserve_drift_budget: float = 0.0
    adaptive_edit_gain: float = 0.0
    adaptive_preserve_gain: float = 0.0
    adaptive_edit_weight_min: float = 0.0
    adaptive_edit_weight_max: float = 10.0
    adaptive_preserve_weight_min: float = 0.0
    adaptive_preserve_weight_max: float = 10.0
    adaptive_preserve_clean_correction_scale: float = 0.0
    adaptive_projection_scale: float = 0.0
    masked_rms_channel_normalize: bool = False
    adaptive_component_control: bool = False
    adaptive_component_threshold: float = 0.5
    adaptive_component_min_pixels: int = 4

    edit_local_target_guidance_scale: float = 0.0
    region_target_transport_scale: float = 0.0
    recolor_clean_projection_scale: float = 0.0

    removal_controller_mode: str = "none"
    edit_operation: str = ""
    removal_fill_scale: float = 0.0
    removal_suppression_scale: float = 0.0
    removal_ring_rec_scale: float = 0.0


@dataclass
class DeceCoreStepInput:
    """All tensors needed by the shared DeCE-RF step.

    Native tensors live in the backend's update layout, e.g. SD3 latent maps or
    FLUX packed latent tokens. Map tensors live in BCHW layout for shared
    clean-estimate energies.
    """

    z_t: torch.Tensor
    x_src: torch.Tensor
    x0_src: torch.Tensor
    x0_tar: torch.Tensor
    v_src: torch.Tensor
    v_tar: torch.Tensor
    v_src_edit: torch.Tensor
    t_scalar: torch.Tensor
    alpha_t: float
    beta_t: float

    x_src_map: torch.Tensor
    x0_src_map: torch.Tensor
    x0_tar_map: torch.Tensor
    base_edit_velocity_map: torch.Tensor
    edit_map: torch.Tensor
    preserve_map: torch.Tensor
    edit_gate: torch.Tensor
    preserve_gate: torch.Tensor
    core_gate: torch.Tensor | None = None

    map_to_native: TensorMapFn = lambda x: x

    target_feature_map: torch.Tensor | None = None
    source_feature_map: torch.Tensor | None = None
    current_rec_feature_map: torch.Tensor | None = None
    source_rec_feature_map: torch.Tensor | None = None
    base_rec_post_gate: torch.Tensor | None = None
    base_edit_post_gate: torch.Tensor | None = None
    x0_local_target: torch.Tensor | None = None
    recolor_projection_target: torch.Tensor | None = None
    recolor_projection_gate: torch.Tensor | None = None


@dataclass
class DeceCoreStepOutput:
    v_rec: torch.Tensor
    v_edit: torch.Tensor
    v_total: torch.Tensor
    edit_terms_norm: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)


def _apply_gate(value: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return value
    gate = gate.to(device=value.device, dtype=value.dtype)
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    if gate.shape[-1] == 1 and value.shape[-1] != 1:
        gate = gate.expand(*value.shape[:-1], value.shape[-1])
    return value * gate


def _masked_rms(
    value: torch.Tensor,
    gate: torch.Tensor | None,
    *,
    normalize_channels: bool = False,
) -> torch.Tensor:
    value = value.float()
    if gate is None:
        return value.square().mean().sqrt()
    gate = gate.float().to(device=value.device)
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    if gate.shape[-1] == 1 and value.shape[-1] != 1:
        gate_for_value = gate.expand(*value.shape[:-1], value.shape[-1])
    else:
        gate_for_value = gate
    denom = gate.sum()
    if normalize_channels and value.ndim >= 2:
        denom = denom * value.shape[1]
    denom = denom.clamp_min(1e-8)
    return ((value.square() * gate_for_value).sum() / denom).sqrt()


def _term_norms(terms: dict[str, torch.Tensor], map_to_native: TensorMapFn) -> dict[str, float]:
    return {
        name: float(map_to_native(value).norm().item())
        for name, value in terms.items()
        if name in {"base", "anchor", "region", "target", "source"}
    }


def _transport_schedule(t_scalar: torch.Tensor, scale: float) -> tuple[float, float, float]:
    t_value = float(t_scalar.detach().item())
    if t_value > 0.65:
        core_beta_base = 1.0
        core_gamma_base = 0.5
        ring_beta_base = 0.25
    elif t_value > 0.35:
        core_beta_base = 0.8
        core_gamma_base = 0.25
        ring_beta_base = 0.35
    else:
        core_beta_base = 0.35
        core_gamma_base = 0.0
        ring_beta_base = 0.15
    core_beta = max(0.0, min(1.0, scale * core_beta_base))
    core_gamma = max(0.0, scale * core_gamma_base)
    ring_beta = max(0.0, min(1.0, scale * ring_beta_base))
    return core_beta, core_gamma, ring_beta


def _label_components_2d(binary: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Label 4-connected components of a 2D binary mask.

    Returns (labels LongTensor HxW with 0=background and 1..n for components,
    n_components). Used to recover the discrete support instances (e.g. each
    animal head) declared by the operation-aware support mask.
    """
    h, w = binary.shape
    b = (binary > 0.5).to(torch.bool).cpu()
    try:
        from scipy.ndimage import label as _sci_label

        lab, n = _sci_label(b.numpy())
        return torch.from_numpy(lab.astype("int64")), int(n)
    except Exception:
        labels = torch.zeros((h, w), dtype=torch.long)
        bb = b.tolist()
        seen = [[False] * w for _ in range(h)]
        cur = 0
        for i in range(h):
            for j in range(w):
                if bb[i][j] and not seen[i][j]:
                    cur += 1
                    stack = [(i, j)]
                    seen[i][j] = True
                    while stack:
                        y, x = stack.pop()
                        labels[y, x] = cur
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and bb[ny][nx] and not seen[ny][nx]:
                                seen[ny][nx] = True
                                stack.append((ny, nx))
        return labels, cur


def compute_component_adaptive_edit_weight_map(
    *,
    current_delta: torch.Tensor,
    target_delta: torch.Tensor,
    target_gap: torch.Tensor,
    edit_map: torch.Tensor,
    preserve_drift: float,
    rmsgap_mode: str,
    rmsgap_dead_zone: float,
    rmsgap_preserve_gate_budget: float,
    edit_target_progress: float,
    edit_target_rms: float,
    edit_gain: float,
    edit_weight_min: float,
    edit_weight_max: float,
    component_threshold: float = 0.5,
    component_min_pixels: int = 4,
    normalize_channels: bool = False,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Build one adaptive edit weight per connected support component.

    The controller uses only clean estimates already available in the current
    RF step. It never copies an edit between instances and adds no model
    forward. Each component follows the same normalized-gap/progress rule as
    the global adaptive controller.
    """

    support = edit_map.float()
    if support.ndim != 4:
        raise ValueError(f"edit_map must be BCHW, got shape={tuple(support.shape)}")
    if support.shape[1] > 1:
        support = support.mean(dim=1, keepdim=True)
    if support.shape[-2:] != target_gap.shape[-2:]:
        support = F.interpolate(support, size=target_gap.shape[-2:], mode="bilinear", align_corners=False)
    support = support.clamp(0.0, 1.0)
    weight_map = torch.ones_like(support)
    component_weights: list[float] = []
    component_count = 0
    active_count = 0
    min_pixels = max(1, int(component_min_pixels))

    for batch_index in range(support.shape[0]):
        labels, count = _label_components_2d(support[batch_index, 0] >= float(component_threshold))
        labels = labels.to(device=support.device)
        for component_index in range(1, count + 1):
            hard_gate = labels == component_index
            if int(hard_gate.sum().item()) < min_pixels:
                continue
            component_count += 1
            hard_gate_4d = hard_gate.to(dtype=support.dtype).view(1, 1, *hard_gate.shape)
            component_gate = support[batch_index : batch_index + 1] * hard_gate_4d
            target_rms_value = float(
                _masked_rms(
                    target_delta[batch_index : batch_index + 1],
                    component_gate,
                    normalize_channels=normalize_channels,
                ).item()
            )
            gap_rms_value = float(
                _masked_rms(
                    target_gap[batch_index : batch_index + 1],
                    component_gate,
                    normalize_channels=normalize_channels,
                ).item()
            )
            if rmsgap_mode == "normgate" and target_rms_value > 1e-6:
                deficit = max(0.0, gap_rms_value / max(target_rms_value, 1e-6) - rmsgap_dead_zone)
                if rmsgap_preserve_gate_budget > 0.0 and preserve_drift >= rmsgap_preserve_gate_budget:
                    deficit = 0.0
            elif edit_target_progress > 0.0:
                current = current_delta[batch_index : batch_index + 1]
                target = target_delta[batch_index : batch_index + 1]
                progress_num = (current * target * component_gate).sum()
                progress_den = (target.square() * component_gate).sum().clamp_min(1e-8)
                progress = float((progress_num / progress_den).detach().item())
                deficit = max(0.0, edit_target_progress - progress)
            elif edit_target_rms > 0.0:
                deficit = max(0.0, edit_target_rms - gap_rms_value)
            else:
                deficit = 0.0
            weight = max(edit_weight_min, min(edit_weight_max, 1.0 + edit_gain * deficit))
            component_weights.append(float(weight))
            if abs(weight - 1.0) > 1e-8:
                active_count += 1
            weight_map[batch_index : batch_index + 1] += (weight - 1.0) * component_gate

    if not component_weights:
        return None, {
            "adaptive_component_count": 0.0,
            "adaptive_component_active_count": 0.0,
            "adaptive_component_weight_min": 1.0,
            "adaptive_component_weight_max": 1.0,
            "adaptive_component_weight_mean": 1.0,
        }
    support_sum = support.sum().clamp_min(1e-8)
    weighted_mean = float(((weight_map * support).sum() / support_sum).item())
    return weight_map, {
        "adaptive_component_count": float(component_count),
        "adaptive_component_active_count": float(active_count),
        "adaptive_component_weight_min": float(min(component_weights)),
        "adaptive_component_weight_max": float(max(component_weights)),
        "adaptive_component_weight_mean": weighted_mean,
    }


def compute_dece_core_step(config: DeceCoreConfig, step: DeceCoreStepInput) -> DeceCoreStepOutput:
    """Compute the shared DeCE-RF edit/preserve velocity components.

    This function intentionally knows nothing about SD3 or FLUX forward APIs.
    Backends provide native tensors plus BCHW map views and a map_to_native
    adapter for the shared energy functions.
    """

    diagnostics: dict[str, float] = {}
    current_delta = step.x0_src - step.x_src.to(torch.float32)
    target_delta = step.x0_tar - step.x_src.to(torch.float32)
    target_gap = step.x0_tar - step.x0_src

    edit_target_rms = float(
        _masked_rms(
            target_delta,
            step.edit_gate,
            normalize_channels=config.masked_rms_channel_normalize,
        ).item()
    )
    edit_gap_rms = float(
        _masked_rms(
            target_gap,
            step.edit_gate,
            normalize_channels=config.masked_rms_channel_normalize,
        ).item()
    )
    preserve_drift = float(
        _masked_rms(
            current_delta,
            step.preserve_gate,
            normalize_channels=config.masked_rms_channel_normalize,
        ).item()
    )
    adaptive_edit_weight = 1.0
    adaptive_preserve_weight = 1.0
    adaptive_projection_norm = 0.0
    adaptive_preserve_clean_correction_norm = 0.0

    if config.adaptive_clean_control:
        edit_progress_num = (_apply_gate(current_delta * target_delta, step.edit_gate)).sum()
        edit_progress_den = (_apply_gate(target_delta.square(), step.edit_gate)).sum().clamp_min(1e-8)
        edit_progress = float((edit_progress_num / edit_progress_den).detach().item())
        if config.adaptive_rmsgap_mode == "normgate" and edit_target_rms > 1e-6:
            normalized_gap = edit_gap_rms / max(edit_target_rms, 1e-6)
            edit_deficit = max(0.0, normalized_gap - config.adaptive_rmsgap_dead_zone)
            if (
                config.adaptive_rmsgap_preserve_gate_budget > 0.0
                and preserve_drift >= config.adaptive_rmsgap_preserve_gate_budget
            ):
                edit_deficit = 0.0
        elif config.adaptive_edit_target_progress > 0.0:
            edit_deficit = max(0.0, config.adaptive_edit_target_progress - edit_progress)
        elif config.adaptive_edit_target_rms > 0.0:
            edit_deficit = max(0.0, config.adaptive_edit_target_rms - edit_gap_rms)
        else:
            edit_deficit = 0.0
        preserve_excess = max(0.0, preserve_drift - config.adaptive_preserve_drift_budget)
        adaptive_edit_weight = 1.0 + config.adaptive_edit_gain * edit_deficit
        adaptive_preserve_weight = 1.0 + config.adaptive_preserve_gain * preserve_excess
        adaptive_edit_weight = max(
            config.adaptive_edit_weight_min,
            min(config.adaptive_edit_weight_max, adaptive_edit_weight),
        )
        adaptive_preserve_weight = max(
            config.adaptive_preserve_weight_min,
            min(config.adaptive_preserve_weight_max, adaptive_preserve_weight),
        )

    rec_terms = reconstruction_velocity_surrogate_total(
        x0_pred=step.x0_src_map,
        x_src=step.x_src_map,
        t_scalar=step.t_scalar,
        M_preserve=step.preserve_map,
        current_feature_map=step.current_rec_feature_map,
        source_feature_map=step.source_rec_feature_map,
        lambda_latent=1.0,
        lambda_struct=0.0,
        lambda_feature=config.struct_guidance_scale,
        velocity_conversion_mode="linear_path",
        velocity_t_min=config.linear_path_t_min,
    )
    v_rec = step.alpha_t * step.map_to_native(rec_terms["total"]).to(device=step.z_t.device, dtype=torch.float32)
    if config.adaptive_clean_control:
        v_rec = adaptive_preserve_weight * v_rec
    v_rec = _apply_gate(v_rec, step.base_rec_post_gate)

    edit_terms = editing_velocity_surrogate_total(
        base_edit_velocity=step.base_edit_velocity_map,
        x0_tar=step.x0_tar_map,
        x0_src=step.x0_src_map,
        t_scalar=step.t_scalar,
        M_edit=step.edit_map,
        target_feature_map=step.target_feature_map,
        source_feature_map=step.source_feature_map,
        lambda_base=config.edit_hedit_guidance_scale,
        lambda_anchor=config.edit_guidance_scale,
        lambda_region=config.edit_region_guidance_scale,
        lambda_target=config.edit_target_guidance_scale,
        lambda_source=config.edit_source_guidance_scale,
        velocity_t_min=config.linear_path_t_min,
    )
    v_edit = step.beta_t * step.map_to_native(edit_terms["total"]).to(device=step.z_t.device, dtype=torch.float32)
    v_edit = _apply_gate(v_edit, step.base_edit_post_gate)
    edit_terms_norm = _term_norms(edit_terms, step.map_to_native)

    local_target_formation_norm = 0.0
    if step.x0_local_target is not None and config.edit_local_target_guidance_scale > 0.0:
        formation_gate = step.edit_gate
        if step.core_gate is not None:
            formation_ring = (step.edit_gate - step.core_gate).clamp(0.0, 1.0)
            formation_gate = (step.core_gate + 0.35 * formation_ring).clamp(0.0, 1.0)
        local_delta = (step.x0_local_target - step.x0_src) * formation_gate
        t_value = float(step.t_scalar.detach().item())
        formation_schedule = max(0.0, min(1.0, (t_value - 0.12) / 0.35))
        v_local_form = (
            step.beta_t
            * config.edit_local_target_guidance_scale
            * formation_schedule
            * clean_delta_to_velocity(local_delta, step.t_scalar, eps=config.linear_path_t_min)
        )
        v_edit = v_edit + v_local_form
        local_target_formation_norm = float(v_local_form.norm().item())

    region_target_transport_norm = 0.0
    region_target_transport_core_beta = 0.0
    region_target_transport_core_gamma = 0.0
    region_target_transport_ring_beta = 0.0
    if config.region_target_transport_scale > 0.0:
        (
            region_target_transport_core_beta,
            region_target_transport_core_gamma,
            region_target_transport_ring_beta,
        ) = _transport_schedule(step.t_scalar, config.region_target_transport_scale)
        transport_core_gate = step.core_gate if step.core_gate is not None else step.edit_gate
        transport_edit_gate = step.edit_gate.clamp(0.0, 1.0)
        transport_core_gate = transport_core_gate.clamp(0.0, 1.0)
        transport_ring_gate = (transport_edit_gate - transport_core_gate).clamp(0.0, 1.0)
        target_delta_velocity = step.v_tar - step.v_src
        region_target_transport = target_delta_velocity * (
            (region_target_transport_core_beta + region_target_transport_core_gamma) * transport_core_gate
            + region_target_transport_ring_beta * transport_ring_gate
        )
        region_target_transport_guidance = step.beta_t * region_target_transport
        v_edit = v_edit + region_target_transport_guidance
        region_target_transport_norm = float(region_target_transport_guidance.norm().item())

    recolor_clean_projection_norm = 0.0
    if step.recolor_projection_target is not None and step.recolor_projection_gate is not None:
        projection_delta = (step.recolor_projection_target - step.x0_src) * step.recolor_projection_gate
        recolor_velocity = clean_delta_to_velocity(
            projection_delta,
            step.t_scalar,
            eps=config.linear_path_t_min,
        )
        recolor_guidance = step.beta_t * config.recolor_clean_projection_scale * recolor_velocity
        v_edit = v_edit + recolor_guidance
        recolor_clean_projection_norm = float(recolor_guidance.norm().item())

    if (
        config.adaptive_clean_control
        and config.adaptive_preserve_clean_correction_scale > 0.0
        and preserve_drift > config.adaptive_preserve_drift_budget
    ):
        preserve_excess = preserve_drift - config.adaptive_preserve_drift_budget
        preserve_correction = clean_delta_to_velocity(-current_delta, step.t_scalar, eps=config.linear_path_t_min)
        preserve_correction = _apply_gate(preserve_correction, step.preserve_gate)
        preserve_correction = config.adaptive_preserve_clean_correction_scale * preserve_excess * preserve_correction
        v_rec = v_rec + preserve_correction
        adaptive_preserve_clean_correction_norm = float(preserve_correction.norm().item())

    adaptive_edit_weight_map = None
    adaptive_component_diagnostics: dict[str, float] = {}
    if (
        config.adaptive_clean_control
        and config.adaptive_component_control
        and config.adaptive_edit_gain > 0.0
    ):
        adaptive_edit_weight_map, adaptive_component_diagnostics = compute_component_adaptive_edit_weight_map(
            current_delta=step.x0_src_map - step.x_src_map,
            target_delta=step.x0_tar_map - step.x_src_map,
            target_gap=step.x0_tar_map - step.x0_src_map,
            edit_map=step.edit_map,
            preserve_drift=preserve_drift,
            rmsgap_mode=config.adaptive_rmsgap_mode,
            rmsgap_dead_zone=config.adaptive_rmsgap_dead_zone,
            rmsgap_preserve_gate_budget=config.adaptive_rmsgap_preserve_gate_budget,
            edit_target_progress=config.adaptive_edit_target_progress,
            edit_target_rms=config.adaptive_edit_target_rms,
            edit_gain=config.adaptive_edit_gain,
            edit_weight_min=config.adaptive_edit_weight_min,
            edit_weight_max=config.adaptive_edit_weight_max,
            component_threshold=config.adaptive_component_threshold,
            component_min_pixels=config.adaptive_component_min_pixels,
            normalize_channels=config.masked_rms_channel_normalize,
        )
        adaptive_edit_weight = adaptive_component_diagnostics["adaptive_component_weight_mean"]

    if config.adaptive_clean_control:
        if adaptive_edit_weight_map is not None:
            w_native = step.map_to_native(adaptive_edit_weight_map).to(
                device=step.z_t.device, dtype=torch.float32
            )
            v_edit = w_native * v_edit
        else:
            v_edit = adaptive_edit_weight * v_edit

    if config.adaptive_clean_control and config.adaptive_projection_scale > 0.0:
        clean_edit_effect = -step.t_scalar * v_edit
        preserve_error_eval = _apply_gate(current_delta, step.preserve_gate)
        clean_effect_eval = _apply_gate(clean_edit_effect, step.preserve_gate)
        conflict_dot = (preserve_error_eval * clean_effect_eval).sum()
        if float(conflict_dot.detach().item()) > 0.0:
            preserve_error_sq = preserve_error_eval.square().sum().clamp_min(1e-8)
            destructive_effect = (conflict_dot / preserve_error_sq) * preserve_error_eval
            scaled = max(0.0, min(1.0, config.adaptive_projection_scale)) * destructive_effect
            clean_edit_effect = clean_edit_effect - scaled
            v_edit = -clean_edit_effect / step.t_scalar.clamp_min(1e-6)
            adaptive_projection_norm = float((scaled / step.t_scalar.clamp_min(1e-6)).norm().item())

    removal_controller_norm = 0.0
    removal_fill_norm = 0.0
    removal_suppression_norm = 0.0
    removal_ring_rec_norm = 0.0
    removal_active = (
        config.removal_controller_mode != "none"
        and config.edit_operation.strip().lower() == "remove_object"
    )
    if removal_active:
        remove_gate = step.edit_gate.to(device=step.z_t.device, dtype=torch.float32).clamp(0.0, 1.0)
        u_fill = clean_delta_to_velocity((step.x0_tar - step.x0_src) * remove_gate, step.t_scalar, eps=config.linear_path_t_min)
        u_suppress = (step.v_tar - step.v_src) * remove_gate
        remove_map = step.edit_map.clamp(0.0, 1.0)
        wide_map = F.max_pool2d(remove_map.float(), kernel_size=7, stride=1, padding=3).clamp(0.0, 1.0)
        ring_gate = step.map_to_native((wide_map - remove_map).clamp(0.0, 1.0)).to(device=step.z_t.device, dtype=torch.float32)
        u_ring_rec = ring_gate * v_rec
        removal_guidance = step.beta_t * (
            config.removal_fill_scale * u_fill
            + config.removal_suppression_scale * u_suppress
            + config.removal_ring_rec_scale * u_ring_rec
        )
        v_edit = v_edit + removal_guidance
        removal_controller_norm = float(removal_guidance.norm().item())
        removal_fill_norm = float((step.beta_t * config.removal_fill_scale * u_fill).norm().item())
        removal_suppression_norm = float((step.beta_t * config.removal_suppression_scale * u_suppress).norm().item())
        removal_ring_rec_norm = float((step.beta_t * config.removal_ring_rec_scale * u_ring_rec).norm().item())

    v_total = step.v_src + v_rec + v_edit
    diagnostics.update(
        {
            "adaptive_edit_weight": float(adaptive_edit_weight),
            "adaptive_preserve_weight": float(adaptive_preserve_weight),
            "adaptive_projection_norm": float(adaptive_projection_norm),
            "adaptive_preserve_clean_correction_norm": float(adaptive_preserve_clean_correction_norm),
            "edit_target_rms": float(edit_target_rms),
            "edit_gap_rms": float(edit_gap_rms),
            "preserve_drift": float(preserve_drift),
            "local_target_formation_norm": float(local_target_formation_norm),
            "region_target_transport_norm": float(region_target_transport_norm),
            "region_target_transport_core_beta": float(region_target_transport_core_beta),
            "region_target_transport_ring_beta": float(region_target_transport_ring_beta),
            "region_target_transport_core_gamma": float(region_target_transport_core_gamma),
            "recolor_clean_projection_norm": float(recolor_clean_projection_norm),
            **adaptive_component_diagnostics,
            "removal_controller_norm": float(removal_controller_norm),
            "removal_fill_norm": float(removal_fill_norm),
            "removal_suppression_norm": float(removal_suppression_norm),
            "removal_ring_rec_norm": float(removal_ring_rec_norm),
        }
    )
    return DeceCoreStepOutput(
        v_rec=v_rec,
        v_edit=v_edit,
        v_total=v_total,
        edit_terms_norm=edit_terms_norm,
        diagnostics=diagnostics,
    )
