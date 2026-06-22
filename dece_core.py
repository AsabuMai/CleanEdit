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


def _masked_rms(value: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
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
    denom = gate.sum().clamp_min(1e-8)
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

    edit_target_rms = float(_masked_rms(target_delta, step.edit_gate).item())
    edit_gap_rms = float(_masked_rms(target_gap, step.edit_gate).item())
    preserve_drift = float(_masked_rms(current_delta, step.preserve_gate).item())
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
        lambda_latent=1.0,
        lambda_struct=0.0,
        lambda_feature=config.struct_guidance_scale,
        velocity_conversion_mode="linear_path",
        velocity_t_min=config.linear_path_t_min,
    )
    v_rec = step.alpha_t * step.map_to_native(rec_terms["total"]).to(device=step.z_t.device, dtype=torch.float32)
    if config.adaptive_clean_control:
        v_rec = adaptive_preserve_weight * v_rec

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

    if config.adaptive_clean_control:
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
