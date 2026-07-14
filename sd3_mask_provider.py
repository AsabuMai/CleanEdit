from __future__ import annotations

from dataclasses import dataclass

import torch

from attention_mask import extract_attention_masks
from generic_support import GenericSupportResult, build_generic_support
from operation_support_v3 import (
    OperationSupportV3Result,
    compute_clean_disagreement,
    compute_velocity_disagreement,
    build_operation_support_v3,
    save_support_debug,
)
from sd3_model_ops import SD3PromptConditioning, calc_cfg_v_sd3
from spatial_masks import load_external_mask_like


@dataclass(frozen=True)
class SD3PromptSupportConfig:
    provider: str
    attention_mask_mode: str
    attention_mask_subject_threshold: float
    attention_mask_core_threshold: float
    new_tokens: str | None
    host_tokens: str | None
    removed_tokens: str | None
    semantic_base_mask_path: str | None
    grounding_method: str | None
    edit_operation: str | None
    mask_policy: str
    relation: str | None
    score: str
    attention_power: float
    disagreement_power: float
    top_percentile: float
    min_area_ratio: float
    max_area_ratio: float
    keep_components: int
    dilate_radius: int
    blur_kernel: int
    temporal_aggregation: str | None
    save_debug_maps: bool
    mask_output_dir: str | None


@dataclass(frozen=True)
class SD3PromptSupportResult:
    support: GenericSupportResult | OperationSupportV3Result
    grounding_mask: torch.Tensor | None


def build_sd3_prompt_support(
    *,
    pipe,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    source: SD3PromptConditioning,
    target: SD3PromptConditioning,
    timesteps,
    t_mid: torch.Tensor,
    v_src_mid: torch.Tensor,
    v_tar_mid: torch.Tensor,
    base_masks: dict[str, torch.Tensor],
    mask_reference: torch.Tensor,
    base_guidance_scale: float,
    tar_guidance_scale: float,
    config: SD3PromptSupportConfig,
) -> SD3PromptSupportResult:
    object_source = base_masks.get("target_changed")
    if config.new_tokens is not None:
        new_masks = extract_attention_masks(
            pipe=pipe,
            x_src=x_src,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            src_prompt_embeds=source.prompt_embeds,
            src_pooled_embeds=source.pooled_prompt_embeds,
            tar_prompt_embeds=target.prompt_embeds,
            tar_pooled_embeds=target.pooled_prompt_embeds,
            t=t_mid,
            mode=config.attention_mask_mode,
            target_token_words=config.new_tokens,
            source_token_words=None,
            subject_threshold=config.attention_mask_subject_threshold,
            core_threshold=config.attention_mask_core_threshold,
        )
        object_source = new_masks.get("target_changed")
    if object_source is None or float(object_source.detach().float().max().item()) <= 1e-6:
        object_source = base_masks.get("combined")

    host_source = None
    if config.host_tokens is not None:
        host_masks = extract_attention_masks(
            pipe=pipe,
            x_src=x_src,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            src_prompt_embeds=source.prompt_embeds,
            src_pooled_embeds=source.pooled_prompt_embeds,
            tar_prompt_embeds=target.prompt_embeds,
            tar_pooled_embeds=target.pooled_prompt_embeds,
            t=t_mid,
            mode=config.attention_mask_mode,
            target_token_words=config.host_tokens,
            source_token_words=config.host_tokens,
            subject_threshold=config.attention_mask_subject_threshold,
            core_threshold=config.attention_mask_core_threshold,
        )
        host_source = torch.maximum(host_masks["source_changed"], host_masks["target_changed"])

    removed_source = None
    if config.removed_tokens is not None:
        removed_masks = extract_attention_masks(
            pipe=pipe,
            x_src=x_src,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            src_prompt_embeds=source.prompt_embeds,
            src_pooled_embeds=source.pooled_prompt_embeds,
            tar_prompt_embeds=target.prompt_embeds,
            tar_pooled_embeds=target.pooled_prompt_embeds,
            t=t_mid,
            mode=config.attention_mask_mode,
            target_token_words=None,
            source_token_words=config.removed_tokens,
            subject_threshold=config.attention_mask_subject_threshold,
            core_threshold=config.attention_mask_core_threshold,
        )
        removed_source = removed_masks["source_changed"]

    grounding_mask = None
    if (
        config.provider == "operation_support_v3"
        and config.semantic_base_mask_path is not None
        and (config.grounding_method or "external_mask").strip().lower() != "none"
    ):
        grounding_mask = load_external_mask_like(mask_reference, config.semantic_base_mask_path)

    if config.provider == "operation_support_v3":
        temporal_mode = (config.temporal_aggregation or "single").strip().lower()
        clean_map_override = None
        velocity_map_override = None
        temporal_step_count = 1
        if temporal_mode in {"mean", "max"}:
            support_indices = sorted(
                {
                    max(0, min(len(timesteps) - 1, len(timesteps) // 4)),
                    max(0, min(len(timesteps) - 1, len(timesteps) // 2)),
                    max(0, min(len(timesteps) - 1, (3 * len(timesteps)) // 4)),
                }
            )
            clean_maps = []
            velocity_maps = []
            mid_index = len(timesteps) // 2
            for support_index in support_indices:
                support_t = timesteps[support_index]
                if support_index == mid_index:
                    v_src_support = v_src_mid
                    v_tar_support = v_tar_mid
                else:
                    v_src_support = calc_cfg_v_sd3(
                        pipe=pipe,
                        latents=x_src,
                        negative_prompt_embeds=source.negative_prompt_embeds,
                        prompt_embeds=source.prompt_embeds,
                        negative_pooled_prompt_embeds=source.negative_pooled_prompt_embeds,
                        pooled_prompt_embeds=source.pooled_prompt_embeds,
                        guidance_scale=base_guidance_scale,
                        t=support_t,
                    )
                    v_tar_support = calc_cfg_v_sd3(
                        pipe=pipe,
                        latents=x_src,
                        negative_prompt_embeds=target.negative_prompt_embeds,
                        prompt_embeds=target.prompt_embeds,
                        negative_pooled_prompt_embeds=target.negative_pooled_prompt_embeds,
                        pooled_prompt_embeds=target.pooled_prompt_embeds,
                        guidance_scale=tar_guidance_scale,
                        t=support_t,
                    )
                clean_maps.append(compute_clean_disagreement(x_src, support_t, v_src_support, v_tar_support))
                velocity_maps.append(compute_velocity_disagreement(v_src_support, v_tar_support))
            temporal_step_count = len(support_indices)
            clean_stack = torch.stack(clean_maps, dim=0)
            velocity_stack = torch.stack(velocity_maps, dim=0)
            if temporal_mode == "max":
                clean_map_override = clean_stack.max(dim=0).values
                velocity_map_override = velocity_stack.max(dim=0).values
            else:
                clean_map_override = clean_stack.mean(dim=0)
                velocity_map_override = velocity_stack.mean(dim=0)
        elif temporal_mode != "single":
            raise ValueError(f"Unsupported support_temporal_aggregation: {config.temporal_aggregation}")
        support = build_operation_support_v3(
            attention_map=object_source,
            x_t=x_src,
            t=t_mid,
            source_velocity=v_src_mid,
            target_velocity=v_tar_mid,
            host_attention_map=host_source,
            removed_attention_map=removed_source,
            grounding_mask=grounding_mask,
            edit_operation=config.edit_operation,
            relation=config.relation,
            candidate=config.score,
            attention_power=config.attention_power,
            disagreement_power=config.disagreement_power,
            top_percentile=config.top_percentile,
            min_area_ratio=config.min_area_ratio,
            max_area_ratio=config.max_area_ratio,
            keep_components=config.keep_components,
            dilate_radius=config.dilate_radius,
            blur_kernel=config.blur_kernel,
            clean_map_override=clean_map_override,
            velocity_map_override=velocity_map_override,
            temporal_aggregation=temporal_mode,
            temporal_steps=temporal_step_count,
            mask_policy=config.mask_policy,
        )
        if config.save_debug_maps and config.mask_output_dir is not None:
            save_support_debug(support, config.mask_output_dir)
    else:
        support = build_generic_support(
            attention_map=object_source,
            x_t=x_src,
            t=t_mid,
            source_velocity=v_src_mid,
            target_velocity=v_tar_mid,
            host_attention_map=host_source,
            removed_attention_map=removed_source,
            edit_operation=config.edit_operation,
            score_mode=config.score,
            attention_power=config.attention_power,
            disagreement_power=config.disagreement_power,
            top_percentile=config.top_percentile,
            min_area_ratio=config.min_area_ratio,
            max_area_ratio=config.max_area_ratio,
            keep_components=config.keep_components,
            dilate_radius=config.dilate_radius,
            blur_kernel=config.blur_kernel,
        )
    return SD3PromptSupportResult(support=support, grounding_mask=grounding_mask)
