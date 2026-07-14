from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class FluxLatentState:
    latents: torch.Tensor
    image_height: int
    image_width: int
    latent_height: int
    latent_width: int
    latent_image_ids: torch.Tensor


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _vae_scale_factor(pipe) -> int:
    scale = getattr(pipe, "vae_scale_factor", None)
    if scale is not None:
        return int(scale)
    block_channels = getattr(getattr(pipe, "vae", None), "config", object()).block_out_channels
    return 2 ** (len(block_channels) - 1)


def _resize_to_flux_multiple(image: Image.Image, max_image_size: int | None = None) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if max_image_size and max(width, height) > max_image_size:
        scale = float(max_image_size) / float(max(width, height))
        width = int(round(width * scale))
        height = int(round(height * scale))
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _pil_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    data = data.unsqueeze(0) * 2.0 - 1.0
    return data.to(device=device, dtype=dtype)


def _vae_encode(pipe, image_tensor: torch.Tensor) -> torch.Tensor:
    posterior = pipe.vae.encode(image_tensor).latent_dist
    latents = posterior.sample()
    config = pipe.vae.config
    shift = float(getattr(config, "shift_factor", 0.0) or 0.0)
    scale = float(getattr(config, "scaling_factor", 1.0) or 1.0)
    return (latents - shift) * scale


def _vae_decode(pipe, latents: torch.Tensor) -> torch.Tensor:
    config = pipe.vae.config
    shift = float(getattr(config, "shift_factor", 0.0) or 0.0)
    scale = float(getattr(config, "scaling_factor", 1.0) or 1.0)
    latents = (latents / scale) + shift
    return pipe.vae.decode(latents, return_dict=False)[0]


def _pack_latents(pipe, latents: torch.Tensor) -> torch.Tensor:
    if hasattr(pipe, "_pack_latents"):
        return pipe._pack_latents(
            latents,
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
            latents.shape[3],
        )
    bsz, channels, height, width = latents.shape
    latents = latents.view(bsz, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(bsz, (height // 2) * (width // 2), channels * 4)


def _unpack_latents(pipe, latents: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    scale = _vae_scale_factor(pipe)
    if hasattr(pipe, "_unpack_latents"):
        return pipe._unpack_latents(latents, image_height, image_width, scale)
    latent_height = image_height // scale
    latent_width = image_width // scale
    bsz, _, channels4 = latents.shape
    channels = channels4 // 4
    latents = latents.view(bsz, latent_height // 2, latent_width // 2, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(bsz, channels, latent_height, latent_width)


def _prepare_image_ids(
    pipe,
    batch_size: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    packed_h = latent_height // 2
    packed_w = latent_width // 2
    if hasattr(pipe, "_prepare_latent_image_ids"):
        return pipe._prepare_latent_image_ids(batch_size, latent_height, latent_width, device, dtype)
    latent_image_ids = torch.zeros(packed_h, packed_w, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = torch.arange(packed_h, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = torch.arange(packed_w, device=device, dtype=dtype)[None, :]
    return latent_image_ids.reshape(packed_h * packed_w, 3)


def encode_flux_prompt(
    pipe,
    prompt: str,
    device: torch.device,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
) -> dict[str, torch.Tensor]:
    encoded = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    values = _as_tuple(encoded)
    if len(values) < 2:
        raise RuntimeError("FluxPipeline.encode_prompt returned an unexpected value")
    result = {
        "prompt_embeds": values[0],
        "pooled_prompt_embeds": values[1],
    }
    if len(values) >= 3 and values[2] is not None:
        result["text_ids"] = values[2]
    elif hasattr(pipe, "_get_clip_prompt_embeds"):
        result["text_ids"] = torch.zeros(values[0].shape[1], 3, device=device, dtype=values[0].dtype)
    else:
        result["text_ids"] = torch.zeros(values[0].shape[1], 3, device=device, dtype=values[0].dtype)
    return result


@torch.no_grad()
def encode_flux_image(
    pipe,
    image_path: str | Path,
    device: torch.device,
    max_image_size: int | None = None,
    dtype: torch.dtype | None = None,
) -> FluxLatentState:
    image = _resize_to_flux_multiple(Image.open(image_path), max_image_size=max_image_size)
    vae_dtype = dtype or next(pipe.vae.parameters()).dtype
    image_tensor = _pil_to_tensor(image, device=device, dtype=vae_dtype)
    latents_4d = _vae_encode(pipe, image_tensor)
    packed = _pack_latents(pipe, latents_4d)
    ids = _prepare_image_ids(
        pipe,
        batch_size=packed.shape[0],
        latent_height=latents_4d.shape[-2],
        latent_width=latents_4d.shape[-1],
        device=device,
        dtype=packed.dtype,
    )
    return FluxLatentState(
        latents=packed,
        image_height=image.height,
        image_width=image.width,
        latent_height=latents_4d.shape[-2],
        latent_width=latents_4d.shape[-1],
        latent_image_ids=ids,
    )


@torch.no_grad()
def decode_flux_latents(pipe, state: FluxLatentState | torch.Tensor, image_height: int | None = None, image_width: int | None = None):
    if isinstance(state, FluxLatentState):
        latents = state.latents
        image_height = state.image_height
        image_width = state.image_width
    else:
        latents = state
        if image_height is None or image_width is None:
            raise ValueError("image_height and image_width are required when decoding a raw tensor")
    latents_4d = _unpack_latents(pipe, latents, int(image_height), int(image_width))
    image = _vae_decode(pipe, latents_4d)
    return pipe.image_processor.postprocess(image)


@torch.no_grad()
def encode_flux_unit_image_tensor(
    pipe,
    image: torch.Tensor,
    image_height: int,
    image_width: int,
    latent_image_ids: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> FluxLatentState:
    """
    Encode a BCHW image tensor in [0, 1] into packed FLUX latent space.
    """
    vae_dtype = dtype or next(pipe.vae.parameters()).dtype
    image = image.to(device=next(pipe.vae.parameters()).device, dtype=vae_dtype).clamp(0.0, 1.0)
    image = image * 2.0 - 1.0
    latents_4d = _vae_encode(pipe, image)
    packed = _pack_latents(pipe, latents_4d)
    return FluxLatentState(
        latents=packed,
        image_height=int(image_height),
        image_width=int(image_width),
        latent_height=latents_4d.shape[-2],
        latent_width=latents_4d.shape[-1],
        latent_image_ids=latent_image_ids,
    )


@torch.no_grad()
def decode_flux_latents_to_unit_tensor(pipe, state: FluxLatentState | torch.Tensor, image_height: int | None = None, image_width: int | None = None) -> torch.Tensor:
    """
    Decode packed FLUX latents to a BCHW image tensor in [0, 1].
    """
    if isinstance(state, FluxLatentState):
        latents = state.latents
        image_height = state.image_height
        image_width = state.image_width
    else:
        latents = state
        if image_height is None or image_width is None:
            raise ValueError("image_height and image_width are required when decoding a raw tensor")
    latents_4d = _unpack_latents(pipe, latents, int(image_height), int(image_width))
    image = _vae_decode(pipe, latents_4d)
    return ((image / 2.0) + 0.5).clamp(0.0, 1.0)


def _expand_timestep(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    return t.to(device=device).expand(batch_size)


def _normalize_flux_timestep(t: torch.Tensor) -> torch.Tensor:
    t_float = t.float()
    if float(t_float.detach().max().item()) > 1.0:
        t_float = t_float / 1000.0
    return t_float


def _normalize_spatial_map(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    flat = value.flatten(1)
    lo = flat.amin(dim=1).view(-1, 1, 1, 1)
    hi = flat.amax(dim=1).view(-1, 1, 1, 1)
    return ((value - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


class _FluxAttentionCaptureBuffer:
    def __init__(self):
        self.cross_maps: dict[int, list[torch.Tensor]] = {}
        self.self_maps: dict[int, list[torch.Tensor]] = {}
        self.original_processors: dict[int, object] = {}

    def add_cross(self, layer_index: int, value: torch.Tensor) -> None:
        self.cross_maps.setdefault(layer_index, []).append(value.detach().float())

    def add_self(self, layer_index: int, value: torch.Tensor) -> None:
        self.self_maps.setdefault(layer_index, []).append(value.detach().float())

    def register(self, transformer, layer_indices: list[int]) -> None:
        for index in layer_indices:
            block = transformer.transformer_blocks[index]
            self.original_processors[index] = block.attn.processor
            block.attn.processor = _FluxCapturingAttnProcessor(self, index)

    def remove_hooks(self, transformer) -> None:
        for index, processor in self.original_processors.items():
            transformer.transformer_blocks[index].attn.processor = processor
        self.original_processors.clear()

    def aggregate_cross(self, packed_h: int, packed_w: int, token_indices: list[int] | None = None) -> torch.Tensor:
        maps = []
        for layer_maps in self.cross_maps.values():
            for value in layer_maps:
                if token_indices:
                    valid = [idx for idx in token_indices if 0 <= idx < value.shape[-1]]
                    if valid:
                        maps.append(value[..., valid].mean(dim=-1))
                        continue
                maps.append(value.mean(dim=-1))
        if not maps:
            return torch.zeros(1, 1, packed_h, packed_w)
        avg = torch.stack(maps, dim=0).mean(dim=0)[0]
        return _normalize_spatial_map(avg.reshape(1, 1, packed_h, packed_w))

    def aggregate_self(self, packed_h: int, packed_w: int) -> torch.Tensor:
        maps = []
        for layer_maps in self.self_maps.values():
            for value in layer_maps:
                maps.append(value.mean(dim=-2))
        if not maps:
            return torch.zeros(1, 1, packed_h, packed_w)
        avg = torch.stack(maps, dim=0).mean(dim=0)[0]
        return _normalize_spatial_map(avg.reshape(1, 1, packed_h, packed_w))


class _FluxCapturingAttnProcessor:
    def __init__(self, store: _FluxAttentionCaptureBuffer, layer_index: int):
        self.store = store
        self.layer_index = int(layer_index)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        image_seq_len = hidden_states.shape[1]
        text_seq_len = 0 if encoder_hidden_states is None else encoder_hidden_states.shape[1]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)
            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)
            query = torch.cat([enc_q, query], dim=2)
            key = torch.cat([enc_k, key], dim=2)
            value = torch.cat([enc_v, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        attn_weight = torch.softmax(
            torch.matmul(query.float(), key.float().transpose(-2, -1)) * (head_dim**-0.5),
            dim=-1,
        ).to(dtype=query.dtype)

        if encoder_hidden_states is not None and text_seq_len > 0:
            image_query_slice = slice(text_seq_len, text_seq_len + image_seq_len)
            text_key_slice = slice(0, text_seq_len)
            image_key_slice = slice(text_seq_len, text_seq_len + image_seq_len)
            cross = attn_weight[:, :, image_query_slice, text_key_slice].mean(dim=1)
            self_attn = attn_weight[:, :, image_query_slice, image_key_slice].mean(dim=1)
            self.store.add_cross(self.layer_index, cross)
            self.store.add_self(self.layer_index, self_attn)

        hidden_states = torch.matmul(attn_weight, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :text_seq_len],
                hidden_states[:, text_seq_len:],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


def flux_token_indices_for_words(
    pipe,
    prompt: str,
    words: list[str] | None,
    max_sequence_length: int = 512,
) -> list[int]:
    if not words:
        return []
    tokenizer = getattr(pipe, "tokenizer_2", None)
    if tokenizer is None:
        return []
    wanted = [word.lower().strip() for word in words if word and word.strip()]
    if not wanted:
        return []
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0].tolist()
    pieces = [tokenizer.decode([tok], skip_special_tokens=True).strip().lower() for tok in token_ids]
    indices: list[int] = []
    for index, piece in enumerate(pieces):
        normalized = re.sub(r"\s+", " ", piece.replace("</w>", "")).strip()
        if not normalized:
            continue
        if any(word in normalized or normalized in word for word in wanted):
            indices.append(index)
    return sorted(set(indices))


def flux_changed_words(src_prompt: str, tar_prompt: str) -> tuple[list[str], list[str]]:
    src_words = re.findall(r"\w+", src_prompt.lower())
    tar_words = re.findall(r"\w+", tar_prompt.lower())
    matcher = difflib.SequenceMatcher(None, src_words, tar_words)
    src_changed: list[str] = []
    tar_changed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            src_changed.extend(src_words[i1:i2])
        if tag in {"replace", "insert"}:
            tar_changed.extend(tar_words[j1:j2])
    return src_changed, tar_changed


_FLUX_EDIT_WORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "in",
    "is",
    "on",
    "the",
    "to",
    "under",
    "wearing",
    "with",
}


def flux_content_edit_words(words: list[str], max_words: int = 3) -> list[str]:
    content = [word for word in words if word and word not in _FLUX_EDIT_WORD_STOPWORDS and len(word) > 1]
    return content[-max_words:] if content else words[-max_words:]


@torch.no_grad()
def extract_flux_prompt_attention_map(
    pipe,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float,
    packed_h: int,
    packed_w: int,
    token_indices: list[int] | None = None,
    layer_indices: list[int] | None = None,
    self_weight: float = 0.0,
) -> torch.Tensor:
    n_blocks = len(pipe.transformer.transformer_blocks)
    if layer_indices is None:
        lo = n_blocks // 4
        hi = max(lo + 1, (3 * n_blocks) // 4)
        layer_indices = list(range(lo, hi))
    transformer_dtype = next(pipe.transformer.parameters()).dtype
    store = _FluxAttentionCaptureBuffer()
    try:
        store.register(pipe.transformer, layer_indices)
        _call_flux_transformer(
            pipe=pipe,
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scale,
            t=t,
        )
    finally:
        store.remove_hooks(pipe.transformer)
    cross = store.aggregate_cross(packed_h, packed_w, token_indices=token_indices).to(device=latents.device)
    if self_weight <= 0.0:
        return cross
    self_map = store.aggregate_self(packed_h, packed_w).to(device=latents.device)
    self_weight = max(0.0, min(1.0, float(self_weight)))
    return _normalize_spatial_map((1.0 - self_weight) * cross + self_weight * self_map).to(dtype=transformer_dtype)


def predict_x0_from_linear_rf_path(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    t_scalar: float | torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(t_scalar):
        t_scalar = torch.tensor(t_scalar, device=x_t.device, dtype=x_t.dtype)
    t_scalar = t_scalar.to(device=x_t.device, dtype=x_t.dtype)
    while t_scalar.ndim < x_t.ndim:
        t_scalar = t_scalar.view(*t_scalar.shape, 1)
    return x_t - t_scalar * v_t


def _call_flux_transformer(
    pipe,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance_scale: float,
    t: torch.Tensor,
) -> torch.Tensor:
    transformer_dtype = next(pipe.transformer.parameters()).dtype
    hidden_states = latents.to(dtype=transformer_dtype)
    timestep = _normalize_flux_timestep(_expand_timestep(t, hidden_states.shape[0], hidden_states.device))
    kwargs = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": prompt_embeds.to(device=hidden_states.device, dtype=transformer_dtype),
        "pooled_projections": pooled_prompt_embeds.to(device=hidden_states.device, dtype=transformer_dtype),
        "txt_ids": text_ids.to(device=hidden_states.device),
        "img_ids": latent_image_ids.to(device=hidden_states.device),
        "joint_attention_kwargs": None,
        "return_dict": False,
    }
    if getattr(pipe.transformer.config, "guidance_embeds", False):
        kwargs["guidance"] = torch.full(
            (hidden_states.shape[0],),
            float(guidance_scale),
            device=hidden_states.device,
            dtype=transformer_dtype,
        )
    return pipe.transformer(**kwargs)[0]


@torch.no_grad()
def calc_cfg_v_flux(
    pipe,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance_scale: float,
    t: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_pooled_prompt_embeds: torch.Tensor | None = None,
    distilled_guidance: float | None = None,
) -> torch.Tensor:
    # When true CFG is requested (negatives provided), the distilled FLUX
    # `guidance` embedding is held at a nominal value and `guidance_scale`
    # becomes the classifier-free extrapolation weight, mirroring the SD3
    # DeCE-RF velocity definition. Without negatives the call falls back to the
    # legacy single distilled forward where `guidance_scale` is the embedding.
    distilled = guidance_scale if distilled_guidance is None else distilled_guidance
    cond = _call_flux_transformer(
        pipe=pipe,
        latents=latents,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        latent_image_ids=latent_image_ids,
        guidance_scale=distilled,
        t=t,
    )
    if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None:
        return cond
    uncond = _call_flux_transformer(
        pipe=pipe,
        latents=latents,
        prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=negative_pooled_prompt_embeds,
        text_ids=text_ids,
        latent_image_ids=latent_image_ids,
        guidance_scale=distilled,
        t=t,
    )
    return uncond + float(guidance_scale) * (cond - uncond)


@torch.no_grad()
def invert_source_flux(
    pipe,
    x_src: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance_scale: float,
    timesteps,
    T_steps: int,
    n_max: int,
    sigmas=None,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
    active_indices = [i for i, _ in enumerate(timesteps) if T_steps - i <= n_max]
    inv_indices = list(reversed(active_indices))
    z_t = x_src.clone().to(torch.float32)
    latents_dtype = x_src.dtype
    if sigmas is None:
        sigma_values = [_normalize_flux_timestep(t).to(torch.float32) for t in timesteps]
        sigma_values.append(torch.zeros_like(sigma_values[-1]))
    else:
        sigma_values = [s.to(device=x_src.device, dtype=torch.float32) for s in sigmas]
    trajectory_by_timestep: dict[int, torch.Tensor] = {}
    if return_trajectory:
        trajectory_by_timestep[0] = z_t.clone().to(latents_dtype)
    for idx in inv_indices:
        t = timesteps[idx]
        sigma_lower = sigma_values[idx + 1]
        sigma_higher = sigma_values[idx]
        v = calc_cfg_v_flux(
            pipe=pipe,
            latents=z_t.to(latents_dtype),
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scale,
            t=t,
        )
        z_t = z_t + (sigma_higher - sigma_lower).to(torch.float32) * v.to(torch.float32)
        if return_trajectory:
            trajectory_by_timestep[int(t.item())] = z_t.clone().to(latents_dtype)
    z_t = z_t.to(latents_dtype)
    if return_trajectory:
        if active_indices:
            trajectory_by_timestep[int(timesteps[active_indices[0]].item())] = z_t
        return z_t, trajectory_by_timestep
    return z_t
