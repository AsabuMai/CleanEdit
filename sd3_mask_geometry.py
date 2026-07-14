from __future__ import annotations

from dataclasses import dataclass

import torch

from spatial_masks import (
    _box_from_mask,
    _box_to_list,
    _clamp_box,
    _conservative_attention_box,
    _expand_box,
    _mask_binary_area_ratio,
    apply_mask_morphology_pair,
    build_object_contact_masks,
    build_recolor_trimap_masks,
    filter_spatial_mask_components,
    load_external_mask_like,
    normalized_box_mask_like,
    translate_spatial_mask,
)


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SD3MaskGeometryConfig:
    object_mask_provider: str
    attention_mask_fallback_threshold: float
    attention_mask_max_area_ratio: float
    auto_box_threshold: float
    edit_mask_component_threshold: float
    edit_mask_keep_components: int
    edit_mask_component_y_min: float | None
    edit_mask_component_y_max: float | None
    edit_mask_shift_y: float
    edit_mask_shift_x: float
    edit_mask_dilate_kernel: int
    edit_mask_erode_kernel: int
    edit_mask_hole_fraction: float
    edit_mask_boundary_noise_scale: float
    edit_mask_smooth_kernel: int
    auto_local_boxes: bool
    auto_edit_pad_x: float
    auto_edit_pad_y: float
    auto_edit_min_width: float
    auto_edit_min_height: float
    auto_source_pad_x: float
    auto_source_pad_y: float
    auto_preserve_pad_x: float
    auto_preserve_start_offset: float
    auto_preserve_height: float
    edit_mask_box: Box | None
    edit_mask_box_mode: str
    edit_mask_exclude_box: Box | None
    source_inject_mask_mode: str
    source_inject_mask_box: Box | None
    final_preserve_box: Box | None
    external_edit_mask_path: str | None
    external_edit_mask_mode: str
    mask_layering_mode: str
    mask_object_threshold: float
    mask_contact_dilate_kernel: int
    mask_contact_scale: float
    mask_contact_edge_threshold: float
    mask_contact_edge_protect_scale: float
    mask_trimap_inner_erode_kernel: int
    mask_trimap_outer_dilate_kernel: int
    mask_trimap_boundary_edit_scale: float
    mask_trimap_boundary_preserve_scale: float
    edit_mask_use_core_as_subject: bool


@dataclass(frozen=True)
class SD3MaskGeometryResult:
    edit_mask: torch.Tensor
    core_mask: torch.Tensor
    preserve_mask: torch.Tensor
    contact_mask: torch.Tensor | None
    structure_edge_mask: torch.Tensor | None
    external_mask: torch.Tensor | None
    subject_local_core: torch.Tensor | None
    auto_anchor_box: Box | None
    mask_area_guard_applied: bool
    mask_area_before_guard: float
    mask_area_guard_box: Box | None
    resolved_edit_mask_box: Box | None
    resolved_edit_mask_exclude_box: Box | None
    resolved_source_inject_mask_box: Box | None
    resolved_final_preserve_box: Box | None


def apply_sd3_mask_geometry(
    *,
    edit_mask: torch.Tensor,
    core_mask: torch.Tensor,
    preserve_mask: torch.Tensor,
    structure_reference: torch.Tensor,
    attention_masks: dict[str, torch.Tensor],
    attention_object: torch.Tensor | None,
    velocity_object: torch.Tensor | None,
    attention_velocity_object: torch.Tensor | None,
    generic_support_score: torch.Tensor | None,
    operation_support_relation: torch.Tensor | None,
    config: SD3MaskGeometryConfig,
) -> SD3MaskGeometryResult:
    M_edit = edit_mask
    M_core = core_mask
    M_preserve = preserve_mask
    M_contact = None
    M_structure_edge = None
    M_subject_local_core = None
    external_mask = None
    auto_anchor_mask = M_core
    auto_anchor_box = None
    mask_area_guard_applied = False
    mask_area_guard_box = None
    resolved_edit_mask_box = config.edit_mask_box
    resolved_edit_mask_exclude_box = config.edit_mask_exclude_box
    resolved_source_inject_mask_box = config.source_inject_mask_box
    resolved_final_preserve_box = config.final_preserve_box

    if (
        config.edit_mask_keep_components > 0
        or config.edit_mask_component_y_min is not None
        or config.edit_mask_component_y_max is not None
    ):
        component_threshold = (
            config.edit_mask_component_threshold if config.edit_mask_component_threshold > 0.0 else 0.5
        )
        M_edit = filter_spatial_mask_components(
            M_edit,
            threshold=component_threshold,
            keep_components=config.edit_mask_keep_components,
            center_y_min=config.edit_mask_component_y_min,
            center_y_max=config.edit_mask_component_y_max,
        ).clamp(0.0, 1.0)
        M_core = filter_spatial_mask_components(
            M_core,
            threshold=component_threshold,
            keep_components=config.edit_mask_keep_components,
            center_y_min=config.edit_mask_component_y_min,
            center_y_max=config.edit_mask_component_y_max,
        ).clamp(0.0, 1.0)
        M_core = torch.minimum(M_core, M_edit)
        M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
        auto_anchor_mask = M_core
        if (
            config.object_mask_provider in {"attention", "attention_velocity", "auto"}
            and attention_object is not None
            and float(M_edit.detach().float().max().item()) <= 1e-6
            and float(attention_object.detach().float().max().item()) > 1e-6
        ):
            M_edit = attention_object.to(dtype=structure_reference.dtype).clamp(0.0, 1.0)
            M_core = M_edit
            M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            auto_anchor_mask = M_core
            print("[mask] component filters removed the attention object; using generic attention object mask")

    if config.edit_mask_shift_y != 0.0 or config.edit_mask_shift_x != 0.0:
        M_edit = translate_spatial_mask(
            M_edit,
            shift_y=config.edit_mask_shift_y,
            shift_x=config.edit_mask_shift_x,
        )
        M_core = translate_spatial_mask(
            M_core,
            shift_y=config.edit_mask_shift_y,
            shift_x=config.edit_mask_shift_x,
        )
        M_core = torch.minimum(M_core, M_edit)
        M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
        auto_anchor_mask = M_core

    M_edit, M_core, M_preserve = apply_mask_morphology_pair(
        M_edit,
        M_core,
        dilate_kernel=config.edit_mask_dilate_kernel,
        erode_kernel=config.edit_mask_erode_kernel,
        hole_fraction=config.edit_mask_hole_fraction,
        boundary_noise_scale=config.edit_mask_boundary_noise_scale,
        smooth_kernel=config.edit_mask_smooth_kernel,
        constrain_after_each=True,
    )
    mask_area_before_guard = _mask_binary_area_ratio(M_edit, threshold=0.5)
    skip_attention_velocity_soft_guard = (
        config.object_mask_provider == "attention_velocity"
        and attention_velocity_object is not None
        and mask_area_before_guard <= 1.30 * config.attention_mask_max_area_ratio
    )
    if (
        config.attention_mask_max_area_ratio > 0.0
        and mask_area_before_guard > config.attention_mask_max_area_ratio
        and not skip_attention_velocity_soft_guard
    ):
        if config.object_mask_provider == "velocity_diff" and velocity_object is not None:
            guard_source = velocity_object
        else:
            guard_source = attention_masks.get("target_changed")
            if guard_source is None or float(guard_source.detach().float().max().item()) <= 1e-6:
                guard_source = attention_masks.get("combined")
        guard_anchor_box = _conservative_attention_box(
            guard_source,
            threshold=config.attention_mask_fallback_threshold,
            fallback=_box_from_mask(M_core, threshold=config.auto_box_threshold),
        )
        if guard_anchor_box is not None:
            mask_area_guard_box = _expand_box(
                guard_anchor_box,
                pad_x=config.auto_edit_pad_x,
                pad_y_top=config.auto_edit_pad_y,
                pad_y_bottom=config.auto_edit_pad_y,
                min_width=0.28,
                min_height=0.12,
            )
            guard_mask = normalized_box_mask_like(M_edit, mask_area_guard_box)
            M_edit = (M_edit * guard_mask).clamp(0.0, 1.0)
            M_core = (M_core * guard_mask).clamp(0.0, 1.0)
            M_core = torch.minimum(M_core, M_edit)
            M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            auto_anchor_mask = guard_mask
            auto_anchor_box = mask_area_guard_box
            mask_area_guard_applied = True
            if resolved_edit_mask_box is None:
                resolved_edit_mask_box = mask_area_guard_box
            print(
                "[mask_guard] "
                f"area={mask_area_before_guard:.2%} "
                f"limit={config.attention_mask_max_area_ratio:.2%} "
                f"box={_box_to_list(mask_area_guard_box)}"
            )

    if config.auto_local_boxes:
        if auto_anchor_box is None:
            auto_anchor_box = _box_from_mask(
                auto_anchor_mask,
                threshold=config.auto_box_threshold,
                fallback=_box_from_mask(M_edit, threshold=config.auto_box_threshold),
            )
        if auto_anchor_box is not None:
            if resolved_edit_mask_box is None:
                resolved_edit_mask_box = _expand_box(
                    auto_anchor_box,
                    pad_x=config.auto_edit_pad_x,
                    pad_y_top=config.auto_edit_pad_y,
                    pad_y_bottom=config.auto_edit_pad_y,
                    min_width=config.auto_edit_min_width,
                    min_height=config.auto_edit_min_height,
                )
            if resolved_source_inject_mask_box is None and config.source_inject_mask_mode == "box":
                resolved_source_inject_mask_box = _expand_box(
                    auto_anchor_box,
                    pad_x=config.auto_source_pad_x,
                    pad_y_top=config.auto_source_pad_y,
                    pad_y_bottom=config.auto_source_pad_y,
                    min_width=0.56,
                    min_height=0.24,
                )
            if resolved_edit_mask_box is None:
                preserve_x0, preserve_y0, preserve_x1, preserve_y1 = auto_anchor_box
            else:
                preserve_x0, preserve_y0, preserve_x1, preserve_y1 = resolved_edit_mask_box
            preserve_width = max(preserve_x1 - preserve_x0, 1e-6)
            px0 = max(0.0, preserve_x0 + 0.15 * preserve_width - config.auto_preserve_pad_x)
            px1 = min(1.0, preserve_x1 - 0.05 * preserve_width + config.auto_preserve_pad_x)
            y1 = preserve_y1
            py0 = min(1.0, y1 + config.auto_preserve_start_offset)
            py1 = min(1.0, py0 + config.auto_preserve_height)
            if py1 > py0:
                preserve_box = _clamp_box((px0, py0, px1, py1))
                if resolved_edit_mask_exclude_box is None:
                    resolved_edit_mask_exclude_box = preserve_box
                if resolved_final_preserve_box is None:
                    resolved_final_preserve_box = preserve_box

    if resolved_edit_mask_box is not None:
        box_mask = normalized_box_mask_like(M_edit, resolved_edit_mask_box)
        if config.edit_mask_box_mode == "replace":
            M_edit = box_mask
            M_core = box_mask
        elif config.edit_mask_box_mode == "intersect":
            M_edit = M_edit * box_mask
            M_core = M_core * box_mask
        elif config.edit_mask_box_mode == "union":
            M_edit = torch.maximum(M_edit, box_mask)
            M_core = torch.maximum(M_core, box_mask)
        else:
            raise ValueError(f"Unsupported edit_mask_box_mode: {config.edit_mask_box_mode}")
        M_core = torch.minimum(M_core, M_edit)
        M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)

    if resolved_edit_mask_exclude_box is not None:
        exclude_mask = normalized_box_mask_like(M_edit, resolved_edit_mask_exclude_box)
        keep_mask = (1.0 - exclude_mask).clamp(0.0, 1.0)
        M_edit = (M_edit * keep_mask).clamp(0.0, 1.0)
        M_core = (M_core * keep_mask).clamp(0.0, 1.0)
        M_core = torch.minimum(M_core, M_edit)
        M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)

    if config.external_edit_mask_path is not None:
        external_mask = load_external_mask_like(M_edit, config.external_edit_mask_path)
        if config.external_edit_mask_mode == "replace":
            M_edit = external_mask
            M_core = external_mask
        elif config.external_edit_mask_mode == "intersect":
            M_edit = M_edit * external_mask
            M_core = M_core * external_mask
        elif config.external_edit_mask_mode == "union":
            M_edit = torch.maximum(M_edit, external_mask)
            M_core = torch.maximum(M_core, external_mask)
        elif config.external_edit_mask_mode == "subject_core":
            local_core = M_edit
            if generic_support_score is not None:
                score_core = generic_support_score.to(dtype=local_core.dtype, device=local_core.device)
                if score_core.shape[-2:] != local_core.shape[-2:]:
                    score_core = torch.nn.functional.interpolate(
                        score_core,
                        size=local_core.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                local_core = torch.maximum(local_core, score_core.clamp(0.0, 1.0))
            if operation_support_relation is not None:
                relation_core = operation_support_relation.to(dtype=local_core.dtype, device=local_core.device)
                if relation_core.shape[-2:] != local_core.shape[-2:]:
                    relation_core = torch.nn.functional.interpolate(
                        relation_core,
                        size=local_core.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                local_core = torch.maximum(local_core, relation_core.clamp(0.0, 1.0))
            M_edit = external_mask
            M_subject_local_core = torch.minimum(local_core, external_mask).clamp(0.0, 1.0)
            M_core = M_subject_local_core
        else:
            raise ValueError(f"Unsupported external_edit_mask_mode: {config.external_edit_mask_mode}")
        M_edit, M_core, M_preserve = apply_mask_morphology_pair(
            M_edit,
            M_core,
            dilate_kernel=config.edit_mask_dilate_kernel,
            erode_kernel=config.edit_mask_erode_kernel,
            hole_fraction=config.edit_mask_hole_fraction,
            boundary_noise_scale=config.edit_mask_boundary_noise_scale,
            smooth_kernel=config.edit_mask_smooth_kernel,
            constrain_after_each=False,
        )

    if config.mask_layering_mode == "object_contact":
        M_edit, M_core, M_contact, M_preserve, M_structure_edge = build_object_contact_masks(
            edit_mask=M_edit,
            core_mask=M_core,
            structure_reference=structure_reference,
            object_threshold=config.mask_object_threshold,
            contact_dilate_kernel=config.mask_contact_dilate_kernel,
            contact_scale=config.mask_contact_scale,
            contact_edge_threshold=config.mask_contact_edge_threshold,
            contact_edge_protect_scale=config.mask_contact_edge_protect_scale,
        )
    elif config.mask_layering_mode == "recolor_trimap":
        M_edit, M_core, M_contact, M_preserve = build_recolor_trimap_masks(
            edit_mask=M_edit,
            core_mask=M_core,
            object_threshold=config.mask_object_threshold,
            inner_erode_kernel=config.mask_trimap_inner_erode_kernel,
            outer_dilate_kernel=config.mask_trimap_outer_dilate_kernel,
            boundary_edit_scale=config.mask_trimap_boundary_edit_scale,
            boundary_preserve_scale=config.mask_trimap_boundary_preserve_scale,
        )
    elif config.mask_layering_mode != "none":
        raise ValueError(f"Unsupported mask_layering_mode: {config.mask_layering_mode}")

    if config.edit_mask_use_core_as_subject:
        M_edit = M_core.clamp(0.0, 1.0)
        M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)

    return SD3MaskGeometryResult(
        edit_mask=M_edit,
        core_mask=M_core,
        preserve_mask=M_preserve,
        contact_mask=M_contact,
        structure_edge_mask=M_structure_edge,
        external_mask=external_mask,
        subject_local_core=M_subject_local_core,
        auto_anchor_box=auto_anchor_box,
        mask_area_guard_applied=mask_area_guard_applied,
        mask_area_before_guard=mask_area_before_guard,
        mask_area_guard_box=mask_area_guard_box,
        resolved_edit_mask_box=resolved_edit_mask_box,
        resolved_edit_mask_exclude_box=resolved_edit_mask_exclude_box,
        resolved_source_inject_mask_box=resolved_source_inject_mask_box,
        resolved_final_preserve_box=resolved_final_preserve_box,
    )
