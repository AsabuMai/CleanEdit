from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from dece_core import DeceCoreConfig, DeceCoreStepInput, DeceCoreStepOutput, compute_dece_core_step
    from energies import clean_delta_to_velocity
except ImportError:  # pragma: no cover - supports direct execution from flux/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dece_core import DeceCoreConfig, DeceCoreStepInput, DeceCoreStepOutput, compute_dece_core_step
    from energies import clean_delta_to_velocity


@dataclass(frozen=True)
class FluxDeceAdapter:
    """Expose FLUX packed latents through the shared CleanEdit core interface."""

    packed_h: int
    packed_w: int

    def packed_to_map(self, packed: torch.Tensor) -> torch.Tensor:
        return packed.transpose(1, 2).reshape(packed.shape[0], packed.shape[2], self.packed_h, self.packed_w)

    def map_to_packed(self, value: torch.Tensor) -> torch.Tensor:
        return value.flatten(2).transpose(1, 2)

    def gate_to_map(self, gate: torch.Tensor | None) -> torch.Tensor | None:
        if gate is None:
            return None
        if gate.ndim == 4:
            return gate
        return gate.transpose(1, 2).reshape(gate.shape[0], gate.shape[2], self.packed_h, self.packed_w)

    def clean_delta_to_native_velocity(
        self,
        clean_delta: torch.Tensor,
        t_scalar: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        return clean_delta_to_velocity(clean_delta, t_scalar, eps=eps)

    def compute_step(
        self,
        config: DeceCoreConfig,
        *,
        z_t: torch.Tensor,
        x_src: torch.Tensor,
        x0_src: torch.Tensor,
        x0_tar: torch.Tensor,
        v_src: torch.Tensor,
        v_tar: torch.Tensor,
        v_src_edit: torch.Tensor,
        t_scalar: torch.Tensor,
        alpha_t: float,
        beta_t: float,
        edit_gate: torch.Tensor,
        preserve_gate: torch.Tensor,
        core_gate: torch.Tensor | None = None,
        target_feature_map: torch.Tensor | None = None,
        source_feature_map: torch.Tensor | None = None,
        x0_local_target: torch.Tensor | None = None,
        recolor_projection_target: torch.Tensor | None = None,
        recolor_projection_gate: torch.Tensor | None = None,
    ) -> DeceCoreStepOutput:
        edit_map = self.gate_to_map(edit_gate)
        preserve_map = self.gate_to_map(preserve_gate)
        if edit_map is None or preserve_map is None:
            raise ValueError("FluxDeceAdapter.compute_step requires edit and preserve gates.")
        return compute_dece_core_step(
            config,
            DeceCoreStepInput(
                z_t=z_t,
                x_src=x_src.to(torch.float32),
                x0_src=x0_src,
                x0_tar=x0_tar,
                v_src=v_src,
                v_tar=v_tar,
                v_src_edit=v_src_edit,
                t_scalar=t_scalar,
                alpha_t=alpha_t,
                beta_t=beta_t,
                x_src_map=self.packed_to_map(x_src.to(torch.float32)),
                x0_src_map=self.packed_to_map(x0_src),
                x0_tar_map=self.packed_to_map(x0_tar),
                base_edit_velocity_map=self.packed_to_map(v_tar - v_src_edit),
                edit_map=edit_map,
                preserve_map=preserve_map,
                edit_gate=edit_gate.to(device=z_t.device, dtype=torch.float32),
                preserve_gate=preserve_gate.to(device=z_t.device, dtype=torch.float32),
                core_gate=None if core_gate is None else core_gate.to(device=z_t.device, dtype=torch.float32),
                map_to_native=self.map_to_packed,
                target_feature_map=target_feature_map,
                source_feature_map=source_feature_map,
                x0_local_target=x0_local_target,
                recolor_projection_target=recolor_projection_target,
                recolor_projection_gate=recolor_projection_gate,
            ),
        )


def build_flux_dece_config(args) -> DeceCoreConfig:
    """Translate FLUX CLI/runtime args into backend-neutral CleanEdit core config."""

    return DeceCoreConfig(
        struct_guidance_scale=float(args.struct_guidance_scale),
        edit_hedit_guidance_scale=float(args.edit_hedit_guidance_scale),
        edit_guidance_scale=float(args.edit_guidance_scale),
        edit_region_guidance_scale=float(args.edit_region_guidance_scale),
        edit_target_guidance_scale=float(args.edit_target_guidance_scale),
        edit_source_guidance_scale=float(args.edit_source_guidance_scale),
        linear_path_t_min=float(args.linear_path_t_min),
        adaptive_clean_control=bool(args.adaptive_clean_control),
        adaptive_rmsgap_mode=args.adaptive_rmsgap_mode,
        adaptive_rmsgap_dead_zone=float(args.adaptive_rmsgap_dead_zone),
        adaptive_rmsgap_preserve_gate_budget=float(args.adaptive_rmsgap_preserve_gate_budget),
        adaptive_edit_target_progress=float(args.adaptive_edit_target_progress),
        adaptive_edit_target_rms=float(args.adaptive_edit_target_rms),
        adaptive_preserve_drift_budget=float(args.adaptive_preserve_drift_budget),
        adaptive_edit_gain=float(args.adaptive_edit_gain),
        adaptive_component_control=bool(args.adaptive_component_control),
        adaptive_component_threshold=float(args.adaptive_component_threshold),
        adaptive_component_min_pixels=int(args.adaptive_component_min_pixels),
        adaptive_preserve_gain=float(args.adaptive_preserve_gain),
        adaptive_edit_weight_min=float(args.adaptive_edit_weight_min),
        adaptive_edit_weight_max=float(args.adaptive_edit_weight_max),
        adaptive_preserve_weight_min=float(args.adaptive_preserve_weight_min),
        adaptive_preserve_weight_max=float(args.adaptive_preserve_weight_max),
        adaptive_preserve_clean_correction_scale=float(args.adaptive_preserve_clean_correction_scale),
        adaptive_projection_scale=float(args.adaptive_projection_scale),
        edit_local_target_guidance_scale=float(args.edit_local_target_guidance_scale),
        region_target_transport_scale=float(args.region_target_transport_scale),
        source_attachment_release_scale=float(args.source_attachment_release_scale),
        source_attachment_release_stop_t=float(args.source_attachment_release_stop_t),
        source_attachment_release_full_t=float(args.source_attachment_release_full_t),
        recolor_clean_projection_scale=float(args.recolor_clean_projection_scale),
        removal_controller_mode=args.removal_controller_mode,
        edit_operation=args.edit_operation or "",
        removal_fill_scale=float(args.removal_fill_scale),
        removal_suppression_scale=float(args.removal_suppression_scale),
        removal_ring_rec_scale=float(args.removal_ring_rec_scale),
    )
