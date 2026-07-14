from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = ROOT / "data/flowedit_compatible_135/manifest_sam_135.json"
REPAIR_ROOT = ROOT / "data/flowedit_compatible_135/repair_allpass_v1"
SD3_OLD = ROOT / "outputs/fe135_subjectpreserve_full_v10_sd3"
FLUX_OLD = ROOT / "outputs/fe135_full_dece_flux_v11c_pcie8_h100"
SD3_FIXED = ROOT / "outputs/fe135_allpass_sd3_v1"
FLUX_FIXED = ROOT / "outputs/fe135_allpass_flux_v1"

SD3_DEFAULT_RECIPE = "allpass_sd3_v1"
FLUX_DEFAULT_RECIPE = "allpass_flux_v1"

P0_BRANCHES = {
    "C5_exact_text_needs_renderer": "text_renderer_v1",
    "C6_short_text_needs_text_aware_refine": "text_renderer_v1",
    "C9_human_pose_identity_material_transfer": "human_pose_lock_v1",
}

CPU_BRANCHES = {
    "C5_exact_text_needs_renderer": "text_renderer_v1",
    "C6_short_text_needs_text_aware_refine": "text_renderer_v1",
    "C7_low_contrast_or_small_recolor": "recolor_local_v1",
    "C8_complex_structure_material_transfer": "material_structure_lock_v1",
    "C9_human_pose_identity_material_transfer": "human_pose_lock_v1",
    "C10_material_artifact_or_overedit": "material_structure_lock_v1",
    "C11_animal_object_material_balance": "material_structure_lock_v1",
}

GPU_CATEGORIES = {
    "C1_replacement_old_object_residue",
    "C2_small_or_compound_accessory",
    "C3_insertion_mask_geometry",
    "C4_insertion_strength_balance",
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

TEXT_COLOR_OVERRIDE = {
    "gas_station": (190, 20, 28),
    "stop": (255, 255, 255),
    "stop_arrow": (255, 255, 255),
    "stop_sticker": (255, 255, 255),
    "free_wifi": (245, 245, 245),
    "luna": (245, 70, 20),
    "groceries": (25, 20, 15),
    "sign": (20, 20, 20),
    "this_must": (245, 245, 245),
}

COLOR_TABLE = {
    "black": (0.02, 0.02, 0.02),
    "white": (0.94, 0.94, 0.90),
    "red": (0.80, 0.04, 0.03),
    "orange": (0.95, 0.34, 0.04),
    "blue": (0.03, 0.16, 0.82),
    "green": (0.04, 0.45, 0.10),
    "yellow": (0.95, 0.78, 0.04),
    "pink": (0.96, 0.36, 0.72),
    "gold": (0.96, 0.68, 0.08),
    "bronze": (0.66, 0.36, 0.16),
}

MATERIAL_TABLE = {
    "gold": (0.96, 0.68, 0.08),
    "golden": (0.96, 0.68, 0.08),
    "bronze": (0.62, 0.34, 0.16),
    "wood": (0.62, 0.34, 0.15),
    "wooden": (0.62, 0.34, 0.15),
    "marble": (0.78, 0.78, 0.74),
    "sand": (0.78, 0.64, 0.38),
    "sculpture": (0.70, 0.64, 0.55),
    "statue": (0.70, 0.64, 0.55),
    "glass": (0.55, 0.82, 0.95),
    "lego": (0.92, 0.58, 0.08),
    "origami": (0.92, 0.92, 0.86),
    "crochet": (0.86, 0.54, 0.26),
    "puppet": (0.72, 0.46, 0.28),
    "toy": (0.93, 0.45, 0.72),
}


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def first_glob(pattern: str) -> Path | None:
    hits = sorted(ROOT.glob(pattern))
    return hits[0] if hits else None


def old_seed_dir(root: Path, key: str) -> Path | None:
    hits = sorted(root.glob(f"{key}/*/seed_10"))
    return hits[0] if hits else None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)


def mask_bbox(mask: Image.Image, threshold: int = 8) -> tuple[int, int, int, int] | None:
    arr = np.asarray(mask)
    ys, xs = np.nonzero(arr > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def dilate_mask(mask: Image.Image, radius: int, blur: float = 0.0) -> Image.Image:
    out = mask
    if radius > 0:
        k = radius * 2 + 1
        out = out.filter(ImageFilter.MaxFilter(k))
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def erode_mask(mask: Image.Image, radius: int, blur: float = 0.0) -> Image.Image:
    out = mask
    if radius > 0:
        k = radius * 2 + 1
        out = out.filter(ImageFilter.MinFilter(k))
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def rect_mask(size: tuple[int, int], box: tuple[int, int, int, int], blur: float = 2.0, ellipse: bool = False) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if ellipse:
        draw.ellipse(box, fill=255)
    else:
        draw.rounded_rectangle(box, radius=max(4, (box[2] - box[0]) // 12), fill=255)
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return mask


def save_mask(path: Path, mask: Image.Image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path)
    return str(path.relative_to(ROOT))


def read_repair_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return {row["key"]: row for row in csv.DictReader(handle)}


def category_defaults(key: str, entry: dict[str, Any], repair_row: dict[str, str] | None) -> tuple[str, str, str]:
    if repair_row:
        return repair_row["primary_category"], repair_row["rerun_priority"], repair_row["severity"]
    return "C0_visual_pass_no_rerun", "P3_no_rerun", "low"


def prompt_override(key: str, entry: dict[str, Any], category: str) -> tuple[str, str]:
    prompt = entry["target_prompt"]
    negative = ""
    if category == "C1_replacement_old_object_residue":
        prompt = prompt + " The cat is wearing only a black top hat, with no crown, no jeweled crown, and no tiara."
        negative = "crown, jeweled crown, tiara, old crown, crown spikes"
    elif category == "C2_small_or_compound_accessory":
        if key == "fe_142_iguana_2_blue_lizard_top_hat":
            prompt = prompt + " The lizard has the same body pose and a clearly visible top hat on its head."
        else:
            prompt = prompt + " The accessory is clearly visible on the head while the animal keeps the same pose."
    elif category == "C3_insertion_mask_geometry":
        if "whipped_cream" in key:
            prompt = prompt + " A distinct swirl of whipped cream sits on top of the milk."
        elif "cherry" in key:
            prompt = prompt + " A single clearly visible cherry sits on top."
        elif "cocktail" in key:
            prompt = prompt + " The cocktail garnish and glass rim are clearly visible."
    elif category == "C4_insertion_strength_balance":
        prompt = prompt + " The added toppings are clearly visible on the top surface."
    elif category == "C7_low_contrast_or_small_recolor":
        prompt = prompt + " The local color change is clear and the same shape is preserved."
    elif category == "C8_complex_structure_material_transfer":
        prompt = prompt.replace("made out of lego bricks", "with lego-brick texture on the same castle silhouette")
        prompt = prompt + " The same towers, windows, roof, silhouette, and camera view are preserved."
    elif category == "C9_human_pose_identity_material_transfer":
        prompt = prompt + " The same person, same pose, same body silhouette, same face position, and same camera view are preserved."
    elif category in {"C10_material_artifact_or_overedit", "C11_animal_object_material_balance"}:
        prompt = prompt + " The same animal or object keeps the same pose, outline, and background."
        negative = "black patches, gray blocks, artifacts, distorted body, changed pose"
    return prompt, negative


def operation_relation(category: str, entry: dict[str, Any]) -> tuple[str, str]:
    if category in {"C1_replacement_old_object_residue", "C8_complex_structure_material_transfer", "C9_human_pose_identity_material_transfer", "C11_animal_object_material_balance"}:
        return "replace", "inside_host"
    if category in {"C2_small_or_compound_accessory", "C3_insertion_mask_geometry", "C4_insertion_strength_balance"}:
        return "add_object", entry.get("sam_support_relation") or "on_surface"
    if category == "C7_low_contrast_or_small_recolor":
        return "recolor", "inside"
    if category in {"C5_exact_text_needs_renderer", "C6_short_text_needs_text_aware_refine"}:
        return "replace", "inside"
    return entry.get("edit_operation") or "auto", entry.get("sam_support_relation") or "auto"


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


def make_repair_mask(key: str, entry: dict[str, Any], category: str) -> tuple[str, dict[str, str]]:
    image_path = resolve(entry["image"])
    assert image_path is not None
    image = Image.open(image_path).convert("RGB")
    size = image.size
    base_path = resolve(entry.get("pp_local_mask") or entry.get("mask"))
    if base_path is None or not base_path.exists():
        base_mask = Image.new("L", size, 255)
    else:
        base_mask = load_mask(base_path, size)

    masks: dict[str, str] = {}
    out_dir = REPAIR_ROOT / "masks"

    if category == "C1_replacement_old_object_residue":
        mask = dilate_mask(base_mask, 12, 2.0)
    elif category == "C2_small_or_compound_accessory":
        mask = dilate_mask(base_mask, 5, 1.4)
        if key == "fe_142_iguana_2_blue_lizard_top_hat":
            body = resolve("data/flowedit_compatible_118/sam_masks_t4/fe_141_iguana_1_green_lizard_lizard_sam.png")
            if body and body.exists():
                body_mask = dilate_mask(load_mask(body, size), 4, 1.2)
                masks["compound_body_mask"] = save_mask(out_dir / f"{key}_body_recolor.png", body_mask)
    elif category == "C3_insertion_mask_geometry":
        bbox = mask_bbox(base_mask)
        if bbox:
            x0, y0, x1, y1 = bbox
            w, h = x1 - x0, y1 - y0
            if "milk" in key or "beer_glass" in key:
                cap_h = max(16, int(h * 0.28))
                y0n = max(0, y0 - int(cap_h * 0.30))
                y1n = min(size[1], y0n + cap_h)
                pad = int(w * 0.08)
                mask = rect_mask(size, (max(0, x0 - pad), y0n, min(size[0], x1 + pad), y1n), 2.0, ellipse=True)
            else:
                cap_h = max(18, int(h * 0.38))
                y1n = min(size[1], y0 + cap_h)
                pad = int(w * 0.10)
                mask = rect_mask(size, (max(0, x0 - pad), max(0, y0 - int(cap_h * 0.2)), min(size[0], x1 + pad), y1n), 2.0)
        else:
            mask = base_mask
    elif category == "C4_insertion_strength_balance":
        mask = dilate_mask(base_mask, 6, 1.6)
    elif category == "C7_low_contrast_or_small_recolor":
        mask = dilate_mask(base_mask, 4, 1.2)
    elif category in {"C8_complex_structure_material_transfer", "C9_human_pose_identity_material_transfer", "C10_material_artifact_or_overedit", "C11_animal_object_material_balance"}:
        mask = dilate_mask(base_mask, 3, 1.6)
    elif category in {"C5_exact_text_needs_renderer", "C6_short_text_needs_text_aware_refine"}:
        mask = dilate_mask(base_mask, 2, 0.8)
    else:
        mask = base_mask

    masks["mask_path"] = save_mask(out_dir / f"{key}_{category}_support.png", mask)
    return masks["mask_path"], masks


def old_to_new_copy(old_dir: Path | None, dest_dir: Path, entry: dict[str, Any], backend: str, registry_item: dict[str, Any]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if old_dir and old_dir.exists():
        for name in ("result.png", "metadata.json", "stats.json"):
            src = old_dir / name
            if src.exists():
                shutil.copy2(src, dest_dir / name)
        src_masks = old_dir / "masks"
        dst_masks = dest_dir / "masks"
        if src_masks.exists() and not dst_masks.exists():
            shutil.copytree(src_masks, dst_masks)
    else:
        source = Image.open(resolve(entry["image"])).convert("RGB")
        source.save(dest_dir / "result.png")
    meta_path = dest_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
        except Exception:
            meta = {}
    meta.update(
        {
            "allpass_v1": True,
            "allpass_backend": backend,
            "allpass_category": registry_item["category"],
            "allpass_priority": registry_item["priority"],
            "allpass_recipe": registry_item[f"{backend}_recipe"],
            "allpass_mask_path": registry_item["mask_path"],
            "allpass_prompt_override": registry_item["prompt_override"],
            "allpass_negative_prompt": registry_item["negative_prompt"],
            "allpass_postprocess_branch": registry_item["postprocess_branch"],
        }
    )
    save_json(meta_path, meta)
    if not (dest_dir / "stats.json").exists():
        save_json(dest_dir / "stats.json", [{"allpass_v1": True, "status": "copied_source_fallback"}])


def find_font(bold: bool = True) -> str | None:
    env_font = os.environ.get("ALLPASS_TEXT_FONT")
    candidates = [
        env_font,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/urw-base35/NimbusSans-Bold.otf" if bold else "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
        "/usr/share/fonts/google-droid/DroidSans-Bold.ttf" if bold else "/usr/share/fonts/google-droid/DroidSans.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


def font_for(
    text: str,
    box: tuple[int, int, int, int],
    multiline: bool = False,
    width_frac: float = 0.88,
    height_frac: float = 0.84,
) -> ImageFont.ImageFont:
    font_path = find_font(True)
    x0, y0, x1, y1 = box
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    lines = text.split("\n") if multiline else [text]
    for size in range(min(96, h), 7, -2):
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        dummy = Image.new("RGB", (4, 4))
        draw = ImageDraw.Draw(dummy)
        widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
        line_box = draw.textbbox((0, 0), "Mg", font=font)
        line_h = line_box[3] - line_box[1]
        if max(widths) <= w * width_frac and line_h * len(lines) <= h * height_frac:
            return font
    return ImageFont.truetype(font_path, 10) if font_path else ImageFont.load_default()


def target_text_for_key(key: str) -> str:
    if "free_wifi_1_free_beer" in key:
        return "FREE\nBEER"
    if "free_wifi_2_free_hugs" in key:
        return "FREE\nHUGS"
    if "groceries" in key:
        word = key.rsplit("_", 1)[-1].upper()
        return f"- {word}\n- BREAD\n- EGGS\n- MILK"
    if "this_must_be_the_place_1_home" in key:
        return "home"
    if "this_must_be_the_place" in key:
        word = key.rsplit("_", 1)[-1].upper()
        return word
    if "_sign_" in key and any(token in key for token in ("cvpr", "eccv", "iccv", "flow")):
        word = key.rsplit("_", 1)[-1].upper()
        return f"{word} IS\nALL YOU\nNEED"
    if "volkswagen" in key:
        return "VW"
    if "luna_1_sol" in key:
        return "SOL"
    if "luna_7_heart" in key:
        return "♥"
    if "luna_6_hi" in key:
        return "hi"
    word = key.rsplit("_", 1)[-1]
    return word.upper()


def text_color_for(key: str, image: Image.Image, mask: Image.Image) -> tuple[int, int, int]:
    for token, color in TEXT_COLOR_OVERRIDE.items():
        if token in key:
            return color
    arr = np.asarray(image).astype(np.float32)
    m = np.asarray(mask) > 32
    if m.any():
        luma = arr[m].mean()
        return (245, 245, 245) if luma < 120 else (25, 25, 25)
    return (25, 25, 25)


def text_stroke_for(key: str, fill: tuple[int, int, int]) -> tuple[int, int, int]:
    if "gas_station" in key:
        return (250, 245, 238)
    if "stop" in key:
        return (105, 20, 20)
    if "free_wifi" in key:
        return (8, 8, 8)
    if "luna" in key:
        return (75, 10, 5)
    if "this_must" in key:
        return (20, 100, 90)
    if "groceries" in key:
        return (236, 214, 184)
    if "_sign_" in key:
        return (245, 245, 245)
    luma = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
    return (0, 0, 0) if luma > 128 else (245, 245, 245)


def text_renderer_params(key: str) -> dict[str, float]:
    params = {
        "pad_x": 0.04,
        "pad_y": 0.08,
        "width_frac": 0.88,
        "height_frac": 0.76,
        "clear_strength": 0.0,
        "stroke_ratio": 0.06,
    }
    if "gas_station" in key:
        params.update({"pad_x": 0.02, "pad_y": 0.04, "width_frac": 0.92, "height_frac": 0.68, "stroke_ratio": 0.05})
    elif "stop" in key:
        params.update({"pad_x": 0.02, "pad_y": 0.04, "width_frac": 0.90, "height_frac": 0.62, "stroke_ratio": 0.07})
    elif "_sign_" in key:
        params.update({"pad_x": 0.05, "pad_y": 0.06, "width_frac": 0.82, "height_frac": 0.70, "stroke_ratio": 0.05})
    elif "this_must" in key:
        params.update({"pad_x": 0.07, "pad_y": 0.06, "width_frac": 0.74, "height_frac": 0.70, "stroke_ratio": 0.05})
    return params


def apply_text_renderer(image: Image.Image, mask: Image.Image, key: str) -> Image.Image:
    out = image.convert("RGB")
    bbox = mask_bbox(mask, 16)
    if not bbox:
        return out
    x0, y0, x1, y1 = bbox
    params = text_renderer_params(key)
    pad_x = max(1, int((x1 - x0) * params["pad_x"]))
    pad_y = max(1, int((y1 - y0) * params["pad_y"]))
    box = (max(0, x0 + pad_x), max(0, y0 + pad_y), min(out.width, x1 - pad_x), min(out.height, y1 - pad_y))
    if "this_must" in key:
        box_h = max(1, box[3] - box[1])
        box = (box[0], box[1], box[2], min(box[3], box[1] + int(box_h * 0.36)))
    text = target_text_for_key(key)
    multiline = "\n" in text
    font = font_for(text, box, multiline, params["width_frac"], params["height_frac"])
    draw = ImageDraw.Draw(out)

    # Only weakly clean the old glyph area. Earlier all-mask cleaning washed out
    # whole signs because these support masks often cover the complete panel.
    if params["clear_strength"] > 0:
        arr = np.asarray(out).copy()
        m = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32) / 255.0
        hard = m > 0.35
    else:
        hard = None
    if hard is not None and hard.any():
        bg = np.median(arr[hard], axis=0)
        strength = float(params["clear_strength"])
        clean = arr.astype(np.float32) * (1.0 - (m[..., None] * strength)) + bg[None, None, :] * (m[..., None] * strength)
        out = Image.fromarray(np.clip(clean, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(out)

    color = text_color_for(key, out, mask)
    stroke = text_stroke_for(key, color)
    stroke_width = max(1, int(getattr(font, "size", 14) * params["stroke_ratio"]))
    lines = text.split("\n")
    line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
    line_h = max((b[3] - b[1] for b in line_boxes), default=10)
    spacing = max(1, line_h // 7)
    total_h = line_h * len(lines) + max(0, len(lines) - 1) * spacing
    cy = (box[1] + box[3] - total_h) // 2
    for line, tb in zip(lines, line_boxes):
        tw = tb[2] - tb[0]
        tx = (box[0] + box[2] - tw) // 2
        draw.text((tx, cy), line, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke)
        cy += line_h + spacing
    return out
def infer_color(key: str, prompt: str) -> tuple[float, float, float]:
    text = f"{key} {prompt}".lower()
    for name, rgb in COLOR_TABLE.items():
        if name in text:
            return rgb
    return (0.12, 0.12, 0.12)


def recolor_preserve_luma(image: Image.Image, mask: Image.Image, target: tuple[float, float, float], strength: float = 0.82) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32) / 255.0
    luma = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    target_arr = np.array(target, dtype=np.float32)[None, None, :]
    target_luma = float(np.dot(np.array(target, dtype=np.float32), np.array([0.299, 0.587, 0.114], dtype=np.float32)))
    scaled = np.clip(target_arr * ((luma[..., None] + 0.06) / max(0.06, target_luma + 0.06)), 0.0, 1.0)
    mixed = arr * (1.0 - strength) + scaled * strength
    out = arr * (1.0 - m[..., None]) + mixed * m[..., None]
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))


def material_rgb(key: str, prompt: str) -> tuple[float, float, float]:
    text = f"{key} {prompt}".lower()
    for name, rgb in MATERIAL_TABLE.items():
        if name in text:
            return rgb
    return (0.70, 0.64, 0.55)


def make_noise_texture(size: tuple[int, int], seed: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    tex = Image.fromarray(np.clip((noise * 28 + 128), 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    return np.asarray(tex).astype(np.float32) / 255.0


def material_stylize(image: Image.Image, mask: Image.Image, key: str, prompt: str, strength: float = 0.72) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255.0
    luma = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    rgb = np.array(material_rgb(key, prompt), dtype=np.float32)
    tex = make_noise_texture(image.size, seed=abs(hash(key)) % (2**32))
    shade = np.clip(luma * 0.75 + tex * 0.25, 0.0, 1.0)
    stylized = np.clip(rgb[None, None, :] * (0.55 + shade[..., None] * 0.85), 0.0, 1.0)

    if "lego" in key.lower() or "lego" in prompt.lower():
        grid = np.zeros_like(luma)
        step = max(8, image.width // 34)
        grid[:, ::step] = 1.0
        grid[:: max(7, step // 2), :] = 1.0
        grid = np.asarray(Image.fromarray((grid * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.35))).astype(np.float32) / 255.0
        stylized = np.clip(stylized * (1.0 - grid[..., None] * 0.45), 0.0, 1.0)
    elif "wood" in key.lower() or "wood" in prompt.lower():
        yy = np.linspace(0, 1, image.height)[:, None]
        grain = 0.08 * np.sin((yy * image.height / 7.0) + tex * 5.0)
        stylized = np.clip(stylized + grain[..., None], 0.0, 1.0)
    elif "marble" in key.lower() or "marble" in prompt.lower():
        stylized = np.clip(stylized + (tex[..., None] - 0.5) * 0.25, 0.0, 1.0)

    mixed = arr * (1.0 - strength) + stylized * strength
    out = arr * (1.0 - m[..., None]) + mixed * m[..., None]
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))


def draw_top_hat(draw: ImageDraw.ImageDraw, cx: float, base_y: float, w: float, h: float) -> None:
    x0, x1 = cx - w / 2, cx + w / 2
    brim_h = h * 0.16
    body_h = h * 0.70
    y1 = base_y
    y0 = y1 - body_h
    draw.ellipse((x0 - w * 0.18, y1 - brim_h, x1 + w * 0.18, y1 + brim_h), fill=(9, 10, 12), outline=(55, 55, 55), width=max(1, int(w * 0.035)))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(2, int(w * 0.08)), fill=(13, 15, 18), outline=(72, 72, 72), width=max(1, int(w * 0.035)))
    draw.rectangle((x0, y1 - h * 0.23, x1, y1 - h * 0.13), fill=(36, 38, 42))
    draw.arc((x0, y0 - brim_h, x1, y0 + brim_h), 180, 360, fill=(105, 105, 105), width=max(1, int(w * 0.035)))


def draw_crown(draw: ImageDraw.ImageDraw, cx: float, base_y: float, w: float, h: float) -> None:
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = base_y - h, base_y
    gold = (218, 164, 44)
    dark = (80, 52, 8)
    pts = [
        (x0, y1),
        (x0 + w * 0.12, y0 + h * 0.32),
        (x0 + w * 0.30, y1),
        (cx, y0),
        (x0 + w * 0.70, y1),
        (x1 - w * 0.12, y0 + h * 0.32),
        (x1, y1),
    ]
    draw.polygon(pts, fill=gold, outline=dark)
    draw.rectangle((x0, y1 - h * 0.22, x1, y1), fill=(190, 124, 30), outline=dark, width=max(1, int(w * 0.035)))
    for px, py in [(x0 + w * 0.12, y0 + h * 0.32), (cx, y0), (x1 - w * 0.12, y0 + h * 0.32)]:
        r = max(2, w * 0.055)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(245, 222, 92), outline=dark)


def apply_accessory_renderer(source: Image.Image, key: str) -> Image.Image:
    scale = 3
    canvas = source.convert("RGB").resize((source.width * scale, source.height * scale), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)

    def sx(x: float) -> float:
        return x * source.width * scale

    def sy(y: float) -> float:
        return y * source.height * scale

    if key == "fe_192_parrots_1_top_hat":
        # Small hats on both heads; keep them much smaller than the diffusion proposal.
        draw_top_hat(draw, sx(0.390), sy(0.455), source.width * 0.072 * scale, source.height * 0.086 * scale)
        draw_top_hat(draw, sx(0.604), sy(0.420), source.width * 0.076 * scale, source.height * 0.090 * scale)
    elif key == "fe_195_parrots2_1_crown":
        draw_crown(draw, sx(0.405), sy(0.265), source.width * 0.074 * scale, source.height * 0.052 * scale)
        draw_crown(draw, sx(0.690), sy(0.365), source.width * 0.074 * scale, source.height * 0.052 * scale)
    else:
        return source.convert("RGB")
    return canvas.resize(source.size, Image.Resampling.LANCZOS)


def apply_cpu_postprocess(dest_dir: Path, entry: dict[str, Any], registry_item: dict[str, Any], backend: str) -> None:
    branch = registry_item["postprocess_branch"]
    if branch in ("none", ""):
        return
    result_path = dest_dir / "result.png"
    if not result_path.exists():
        return
    image = Image.open(result_path).convert("RGB")
    source = Image.open(resolve(entry["image"])).convert("RGB").resize(image.size, Image.Resampling.BILINEAR)
    mask_path = resolve(registry_item["mask_path"])
    if mask_path is None or not mask_path.exists():
        return
    mask = load_mask(mask_path, image.size)

    if branch == "text_renderer_v1":
        fixed = apply_text_renderer(image, mask, entry["key"])
    elif branch == "human_pose_lock_v1":
        fixed = material_stylize(source, mask, entry["key"], registry_item["prompt_override"], strength=0.82)
    elif branch == "accessory_renderer_v1":
        fixed = apply_accessory_renderer(source, entry["key"])
    elif branch == "compound_recolor_v1":
        body_path = registry_item.get("mask_extra", {}).get("compound_body_mask")
        body_mask_path = resolve(body_path)
        body_mask = load_mask(body_mask_path, image.size) if body_mask_path and body_mask_path.exists() else mask
        fixed = recolor_preserve_luma(image, body_mask, (0.05, 0.24, 0.62), strength=0.68)
    elif branch == "recolor_local_v1":
        fixed = recolor_preserve_luma(source, mask, infer_color(entry["key"], registry_item["prompt_override"]), strength=0.86)
    elif branch == "material_structure_lock_v1":
        fixed = material_stylize(source, mask, entry["key"], registry_item["prompt_override"], strength=0.76)
    else:
        return
    fixed.save(result_path)

    meta_path = dest_dir / "metadata.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    meta["allpass_cpu_postprocess_applied"] = branch
    meta["allpass_cpu_postprocess_source"] = "source_image" if branch != "text_renderer_v1" else "old_result_image"
    meta["evaluation_eligible"] = False
    meta["paper_use"] = False
    save_json(meta_path, meta)


def registry_item_for(entry: dict[str, Any], repair_row: dict[str, str] | None) -> dict[str, Any]:
    key = entry["key"]
    category, priority, severity = category_defaults(key, entry, repair_row)
    mask_path, mask_extra = make_repair_mask(key, entry, category)
    prompt, negative = prompt_override(key, entry, category)
    operation, relation = operation_relation(category, entry)
    relation = normalize_relation(relation, operation, key)
    if key == "fe_142_iguana_2_blue_lizard_top_hat":
        post_branch = "compound_recolor_v1"
    elif key in {"fe_192_parrots_1_top_hat", "fe_195_parrots2_1_crown"}:
        # Main-comparison accessories must be produced by model inference.
        post_branch = "none"
    else:
        post_branch = CPU_BRANCHES.get(category, "none")
    # These T3 text families already read well in the old SD3/FLUX results. A
    # renderer pass makes them look artificial, so keep the old result there and
    # reserve rendering for the failing long-range/sign/stop-arrow cases.
    if post_branch == "text_renderer_v1" and (
        any(token in key for token in ("free_wifi", "groceries", "luna", "stop_sticker"))
        or ("this_must_be_the_place" in key and "2_cvpr" not in key)
    ):
        post_branch = "none"
    item = {
        "key": key,
        "category": category,
        "priority": priority,
        "severity": severity,
        "sd3_recipe": SD3_DEFAULT_RECIPE if category != "C0_visual_pass_no_rerun" else "copy_v10_sd3",
        "flux_recipe": FLUX_DEFAULT_RECIPE if category != "C0_visual_pass_no_rerun" else "copy_v11c_flux",
        "mask_path": mask_path,
        "mask_extra": mask_extra,
        "prompt_override": prompt,
        "negative_prompt": negative,
        "postprocess_branch": post_branch,
        "edit_operation": operation,
        "support_relation": relation,
        "requires_gpu_rerun": category in GPU_CATEGORIES,
    }
    return item


def manifest_entry_with_registry(entry: dict[str, Any], item: dict[str, Any], backend: str) -> dict[str, Any]:
    out = dict(entry)
    out["target_prompt_original"] = entry["target_prompt"]
    out["target_prompt"] = item["prompt_override"]
    out["negative_prompt"] = item["negative_prompt"]
    out["pp_local_mask_original"] = entry.get("pp_local_mask")
    out["pp_local_mask"] = item["mask_path"]
    out["allpass_category"] = item["category"]
    out["allpass_priority"] = item["priority"]
    out["allpass_recipe"] = item[f"{backend}_recipe"]
    out["allpass_postprocess_branch"] = item["postprocess_branch"]
    out["edit_operation"] = item["edit_operation"]
    out["sam_support_relation"] = item["support_relation"]
    return out


def make_outputs(manifest: list[dict[str, Any]], registry: dict[str, Any], backend: str, apply_cpu: bool) -> None:
    fixed_root = SD3_FIXED if backend == "sd3" else FLUX_FIXED
    old_root = SD3_OLD if backend == "sd3" else FLUX_OLD
    for entry in manifest:
        key = entry["key"]
        item = registry[key]
        recipe = item[f"{backend}_recipe"]
        dest = fixed_root / key / recipe / "seed_10"
        old_dir = old_seed_dir(old_root, key)
        old_to_new_copy(old_dir, dest, entry, backend, item)
        if apply_cpu:
            apply_cpu_postprocess(dest, entry, item, backend)


def write_contact_sheet(manifest: list[dict[str, Any]], registry: dict[str, Any]) -> None:
    out = REPAIR_ROOT / "debug" / "allpass_cpu_preview.jpg"
    keys = [e["key"] for e in manifest if registry[e["key"]]["postprocess_branch"] != "none"][:80]
    if not keys:
        return
    task_map = {e["key"]: e for e in manifest}
    thumb = 92
    pad = 8
    left = 250
    row_h = thumb + 36
    cols = ["source", "old_sd3", "fixed_sd3", "old_flux", "fixed_flux"]
    canvas = Image.new("RGB", (left + len(cols) * (thumb + pad) + pad, 28 + len(keys) * row_h + pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 8), "allpass_v1 CPU branch preview", fill=(0, 0, 0), font=font)
    for j, col in enumerate(cols):
        draw.text((left + j * (thumb + pad), 8), col, fill=(0, 0, 0), font=font)
    for i, key in enumerate(keys):
        entry = task_map[key]
        item = registry[key]
        y = 28 + i * row_h
        draw.text((pad, y + 5), key[:44], fill=(0, 0, 0), font=font)
        draw.text((pad, y + 20), item["category"], fill=(80, 80, 80), font=font)
        paths = [
            resolve(entry["image"]),
            old_seed_dir(SD3_OLD, key) / "result.png" if old_seed_dir(SD3_OLD, key) else None,
            SD3_FIXED / key / item["sd3_recipe"] / "seed_10" / "result.png",
            old_seed_dir(FLUX_OLD, key) / "result.png" if old_seed_dir(FLUX_OLD, key) else None,
            FLUX_FIXED / key / item["flux_recipe"] / "seed_10" / "result.png",
        ]
        for j, path in enumerate(paths):
            x = left + j * (thumb + pad)
            box = (x, y + 32, x + thumb, y + 32 + thumb)
            draw.rectangle(box, fill=(244, 244, 244), outline=(210, 210, 210))
            if path and path.exists():
                im = Image.open(path).convert("RGB")
                im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                canvas.paste(im, (x + (thumb - im.width) // 2, y + 32 + (thumb - im.height) // 2))
            draw.rectangle(box, outline=(190, 190, 190))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-csv", required=True)
    parser.add_argument("--no-output-copy", action="store_true")
    parser.add_argument("--no-cpu-postprocess", action="store_true")
    args = parser.parse_args()

    REPAIR_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = load_json(MANIFEST)
    repair_rows = read_repair_csv(resolve(args.repair_csv) or Path(args.repair_csv))

    registry: dict[str, Any] = {}
    sd3_manifest = []
    flux_manifest = []
    for entry in manifest:
        row = repair_rows.get(entry["key"])
        item = registry_item_for(entry, row)
        registry[entry["key"]] = item
        sd3_manifest.append(manifest_entry_with_registry(entry, item, "sd3"))
        flux_manifest.append(manifest_entry_with_registry(entry, item, "flux"))

    save_json(REPAIR_ROOT / "registry.json", registry)
    save_json(REPAIR_ROOT / "manifest_allpass_sd3_v1.json", sd3_manifest)
    save_json(REPAIR_ROOT / "manifest_allpass_flux_v1.json", flux_manifest)

    if not args.no_output_copy:
        make_outputs(manifest, registry, "sd3", apply_cpu=not args.no_cpu_postprocess)
        make_outputs(manifest, registry, "flux", apply_cpu=not args.no_cpu_postprocess)
        write_contact_sheet(manifest, registry)

    counts: dict[str, int] = {}
    for item in registry.values():
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    save_json(REPAIR_ROOT / "summary.json", {"num_tasks": len(manifest), "category_counts": counts})
    print(f"wrote {REPAIR_ROOT}")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
