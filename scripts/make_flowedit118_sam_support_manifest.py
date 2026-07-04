from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

PROJ = Path("/cluster/users/grad/2025/25t8103/project")
SCRIPTS = PROJ / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_semantic_mask as msm

MANIFEST = Path(os.environ.get("MANIFEST", PROJ / "data/flowedit_compatible_118/manifest.json"))
OUT_DIR = Path(os.environ.get("OUT_DIR", PROJ / "data/flowedit_compatible_118/sam_support_masks_all"))
OUT_MANIFEST = Path(
    os.environ.get("OUT_MANIFEST", PROJ / "data/flowedit_compatible_118/manifest_sam_all.json")
)
CONTACT_SHEET = OUT_DIR / "_contact_sheet.jpg"
AUDIT_CSV = OUT_DIR / "_audit.csv"

GROUNDING_MODEL = os.environ.get("GROUNDING_MODEL", "IDEA-Research/grounding-dino-base")
SAM_MODEL = os.environ.get("SAM_MODEL", "facebook/sam-vit-base")
BOX_THRESHOLD = float(os.environ.get("BOX_THRESHOLD", "0.18"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.15"))
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "512"))
MASK_MODE = os.environ.get("MASK_MODE", "sam_box_intersect")
DEVICE = os.environ.get("DEVICE", "cuda:0")
KEEP_ALL_BOXES = os.environ.get("KEEP_ALL_BOXES", "0") == "1"
# Score gate for KEEP_ALL_BOXES: drop grounding boxes below
# max(BOX_SCORE_FLOOR, BOX_SCORE_REL * top_score) so diffuse low-confidence
# boxes cannot balloon the union into a near-global mask.
BOX_SCORE_FLOOR = float(os.environ.get("BOX_SCORE_FLOOR", "0.28"))
BOX_SCORE_REL = float(os.environ.get("BOX_SCORE_REL", "0.5"))
# Apply the support relation (top_center band etc.) per anchor component
# instead of once on the union bbox, so every instance gets its own band.
PER_INSTANCE_RELATION = os.environ.get("PER_INSTANCE_RELATION", "0") == "1"
MIN_INSTANCE_AREA = float(os.environ.get("MIN_INSTANCE_AREA", "0.003"))
# Replace-type accessory edits (crown -> top hat) must COVER the accessory
# being replaced, or its pixels survive under the new object as residue.
# Ground the source-prompt words that disappear from the target prompt and
# union their (accessory-sized) masks into the support band.
ACCESSORY_COVER_REMOVED = os.environ.get("ACCESSORY_COVER_REMOVED", "0") == "1"
REMOVED_MAX_AREA = float(os.environ.get("REMOVED_MAX_AREA", "0.18"))

_PROMPT_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "on", "in", "at", "of", "and",
    "with", "to", "it", "its", "his", "her", "their", "this", "that", "placed",
    "wearing", "has", "have", "very", "there",
}


def removed_prompt_tokens(item: dict) -> list[str]:
    import re as _re

    def words(text: str) -> list[str]:
        return [w for w in _re.findall(r"[a-z]+", (text or "").lower()) if w not in _PROMPT_STOPWORDS]

    source = words(item.get("source_prompt", ""))
    target = set(words(item.get("target_prompt", "")))
    seen: set[str] = set()
    removed: list[str] = []
    for w in source:
        if w not in target and w not in seen:
            seen.add(w)
            removed.append(w)
    return removed


IMAGE_STEM_PHRASE_FIX = {
    "bus": ["front of the van", "yellow van", "van"],
    "free_wifi": ["black board", "board", "sign"],
    "gas_station": ["gas station sign", "sign"],
    "groceries": ["brown paper", "paper", "grocery list"],
    "luna": ["neon sign", "sign"],
    "sign": ["billboard", "sign"],
    "stop": ["stop sign", "sign"],
    "stop_arrow": ["stop sign", "sign"],
    "stop_sticker": ["sticker", "stop sticker", "sign"],
    "this_must_be_the_place": ["neon sign", "sign"],
    "cat_and_dog": ["dog and cat", "dog"],
    "parrots": ["parrots", "parrot"],
    "parrots2": ["parrots", "parrot"],
    "penguins": ["penguins", "penguin"],
    "rocks": ["rocks", "stack of rocks"],
    "yellow_bulldog": ["dog figurine", "dog"],
}

KEY_PHRASE_FIX = {
    "fe_017_boat_silhouette_1_sailboat_white_sails_red_hull": ["sailboat", "boat"],
    "fe_024_bus_2_volkswagen_logo": ["front of the van", "yellow van", "van"],
    "fe_027_butterflies_1_yellow": ["butterflies", "butterfly"],
    "fe_084_cupcake_2_red_velvet": ["cupcake"],
    "fe_114_flowers_1_orange_yellow_white": ["flowers", "bouquet"],
    "fe_115_flowers_2_blue_purple_white": ["flowers", "bouquet"],
    "fe_221_rocks_6_colorful_wooden_blocks": ["rocks", "stack of rocks"],
}


def norm_phrase(value: str | None) -> str:
    if not value:
        return ""
    return value.replace(",", " ").replace("_", " ").strip().lower()


def unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = norm_phrase(item)
        if item and item not in seen and item != "object":
            seen.add(item)
            out.append(item)
    return out


def split_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    return unique([x.strip() for x in value.split(",")])


def phrase_candidates(item: dict) -> list[str]:
    key = item["key"]
    stem = Path(item["image"]).stem
    family = item.get("family_label", "")
    host = split_tokens(item.get("host_tokens"))
    amap = item.get("pp_aspect_mapping") or {}
    map_hosts = list(amap.keys())
    phrases: list[str] = []
    phrases += KEY_PHRASE_FIX.get(key, [])
    phrases += IMAGE_STEM_PHRASE_FIX.get(stem, [])
    phrases += host
    phrases += map_hosts
    if family == "T1_attached_accessory":
        phrases += [f"{p} head" for p in host + map_hosts]
        phrases += ["head"]
    elif family == "T2_container_insertion":
        phrases += ["container", "glass", "plate", "cake", "bowl"]
    elif family == "T3_surface_decal":
        phrases += ["sign", "board", "paper", "billboard", "sticker"]
    elif family in {"T4_local_recolor", "T5_same_color_material"}:
        phrases += ["object"]
    return unique(phrases)


def support_plan(item: dict) -> dict[str, object]:
    family = item.get("family_label", "")
    new = set(split_tokens(item.get("new_tokens")))
    if family == "T1_attached_accessory":
        relation = "top_center"
        if new & {"sunglasses", "glasses", "goggles"}:
            relation = "front_glasses_auto"
        return {
            "relation": relation,
            "expand_x": 0.08,
            "expand_y": 0.02,
            "band_ratio": 0.42,
            "overlap_ratio": 0.08,
            "dilate": 5,
            "blur": 5,
            "max_area": 0.95,
        }
    if family == "T2_container_insertion":
        return {
            "relation": "inside",
            "expand_x": 0.0,
            "expand_y": 0.0,
            "band_ratio": 0.55,
            "overlap_ratio": 0.20,
            "dilate": 7,
            "blur": 5,
            "max_area": 0.90,
        }
    if family == "T3_surface_decal":
        return {
            "relation": "inside",
            "expand_x": 0.0,
            "expand_y": 0.0,
            "band_ratio": 0.55,
            "overlap_ratio": 0.20,
            "dilate": 5,
            "blur": 3,
            "max_area": 0.85,
        }
    if family == "T4_local_recolor":
        return {
            "relation": "inside",
            "expand_x": 0.0,
            "expand_y": 0.0,
            "band_ratio": 0.55,
            "overlap_ratio": 0.20,
            "dilate": 7,
            "blur": 5,
            "max_area": 0.90,
        }
    return {
        "relation": "inside",
        "expand_x": 0.0,
        "expand_y": 0.0,
        "band_ratio": 0.55,
        "overlap_ratio": 0.20,
        "dilate": 5,
        "blur": 5,
        "max_area": 0.95,
    }


class GroundedSAM:
    def __init__(self) -> None:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor, SamModel, SamProcessor

        self.device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")
        self.gd_processor = GroundingDinoProcessor.from_pretrained(GROUNDING_MODEL, local_files_only=True)
        self.gd_model = GroundingDinoForObjectDetection.from_pretrained(
            GROUNDING_MODEL, local_files_only=True
        ).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(SAM_MODEL, local_files_only=True)
        self.sam_model = SamModel.from_pretrained(SAM_MODEL, local_files_only=True).to(self.device)
        self.gd_model.eval()
        self.sam_model.eval()
        print("device", self.device, "grounding", GROUNDING_MODEL, "sam", SAM_MODEL, flush=True)

    def mask(self, image: Image.Image, phrase: str, max_area: float) -> tuple[np.ndarray, dict[str, object]]:
        prompt = phrase if phrase.endswith(".") else f"{phrase}."
        gd_inputs = self.gd_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gd_outputs = self.gd_model(**gd_inputs)
        results = self.gd_processor.post_process_grounded_object_detection(
            gd_outputs,
            gd_inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels")
        if boxes is None or len(boxes) == 0:
            raise RuntimeError(f"no boxes for {phrase!r}")
        width, height = image.size
        if max_area > 0.0:
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
            area = (wh[:, 0] * wh[:, 1]) / float(max(1, width * height))
            keep = area <= float(max_area)
            if bool(keep.any().item()):
                boxes = boxes[keep]
                scores = scores[keep] if scores is not None else scores
                if labels is not None:
                    labels = [label for label, flag in zip(labels, keep.detach().cpu().tolist()) if flag]
        if scores is not None and len(boxes) > 1 and not KEEP_ALL_BOXES:
            idx = int(torch.argmax(scores).detach().cpu().item())
            boxes = boxes[idx : idx + 1]
            scores = scores[idx : idx + 1]
            if labels is not None:
                labels = [labels[idx]]
        elif scores is not None and len(boxes) > 1 and KEEP_ALL_BOXES:
            floor = max(BOX_SCORE_FLOOR, BOX_SCORE_REL * float(scores.max().detach().cpu().item()))
            keep = scores >= floor
            if bool(keep.any().item()):
                boxes = boxes[keep]
                scores = scores[keep]
                if labels is not None:
                    labels = [label for label, flag in zip(labels, keep.detach().cpu().tolist()) if flag]

        box_mask = np.zeros((height, width), dtype=np.float32)
        for box in boxes.detach().cpu().tolist():
            x0, y0, x1, y1 = box
            x0 = max(0, min(width, int(np.floor(x0))))
            x1 = max(0, min(width, int(np.ceil(x1))))
            y0 = max(0, min(height, int(np.floor(y0))))
            y1 = max(0, min(height, int(np.ceil(y1))))
            if x1 > x0 and y1 > y0:
                box_mask[y0:y1, x0:x1] = 1.0
        if MASK_MODE == "box":
            union = box_mask
        else:
            sam_inputs = self.sam_processor(
                image,
                input_boxes=[boxes.detach().cpu().tolist()],
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                sam_outputs = self.sam_model(**sam_inputs)
            masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.detach().cpu(),
                sam_inputs["original_sizes"].detach().cpu(),
                sam_inputs["reshaped_input_sizes"].detach().cpu(),
            )[0].float()
            if masks.ndim == 4:
                masks = masks.max(dim=1).values
            union = masks.max(dim=0).values.clamp(0.0, 1.0).numpy()
            if MASK_MODE == "sam_box_intersect":
                union = union * box_mask
        meta = {
            "phrase": phrase,
            "grounding_model": GROUNDING_MODEL,
            "sam_model": SAM_MODEL,
            "mask_mode": MASK_MODE,
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
            "max_box_area_ratio": max_area,
            "num_boxes": int(len(boxes)),
            "boxes_xyxy": [[float(v) for v in box] for box in boxes.detach().cpu().tolist()],
            "scores": [] if scores is None else [float(v) for v in scores.detach().cpu().tolist()],
            "labels": [] if labels is None else [str(v) for v in labels],
        }
        return union, meta


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype("uint8"), mode="L").save(path)


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (255, 60, 20, 0))
    alpha = Image.fromarray((np.clip(mask, 0.0, 1.0) * 160.0).round().astype("uint8"), mode="L")
    red.putalpha(alpha)
    return Image.alpha_composite(base, red).convert("RGB")


def build_contact(items: list[dict]) -> None:
    cells = []
    for item in items[:118]:
        support = Path(item.get("pp_local_mask", ""))
        if not support.is_absolute():
            support = PROJ / support
        if not support.exists():
            continue
        img = Image.open(item["image"]).convert("RGB")
        mask = np.asarray(Image.open(support).convert("L"), dtype=np.float32) / 255.0
        ov = overlay(img.resize((160, 160)), cv2.resize(mask, (160, 160), interpolation=cv2.INTER_LINEAR))
        cells.append((item["key"], ov))
    cols = 8
    label_h = 24
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 160, rows * (160 + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (key, img) in enumerate(cells):
        x = (idx % cols) * 160
        y = (idx // cols) * (160 + label_h)
        sheet.paste(img, (x, y + label_h))
        draw.text((x + 3, y + 5), key[:23], fill=(0, 0, 0))
    CONTACT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT_SHEET, quality=92)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = json.load(MANIFEST.open(encoding="utf-8"))
    sam = GroundedSAM()
    out_items = []
    audit_rows = [
        "idx,key,family,family_label,status,phrase,relation,anchor_area,support_area,support_mask,anchor_mask,error"
    ]
    for idx, item in enumerate(items):
        key = item["key"]
        plan = support_plan(item)
        support_path = OUT_DIR / f"{key}_support.png"
        anchor_path = OUT_DIR / f"{key}_anchor.png"
        meta_path = OUT_DIR / f"{key}.json"
        status = "ok"
        error = ""
        chosen_phrase = ""
        anchor_area = 0.0
        support_area = 0.0
        try:
            image = msm._preprocess_image(item["image"], MAX_IMAGE_SIZE)
            anchor = None
            meta = None
            errors = []
            for phrase in phrase_candidates(item):
                try:
                    anchor, meta = sam.mask(image, phrase, float(plan["max_area"]))
                    chosen_phrase = phrase
                    break
                except Exception as exc:
                    errors.append(f"{phrase}:{exc}")
            if anchor is None or meta is None:
                raise RuntimeError("; ".join(errors[-5:]))

            def relation_support(anchor_part: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
                return msm.support_from_anchor_mask(
                    anchor_part,
                    str(plan["relation"]),
                    image_rgb=np.asarray(image.convert("RGB"), dtype=np.uint8),
                    threshold=0.2,
                    expand_x=float(plan["expand_x"]),
                    expand_y=float(plan["expand_y"]),
                    band_ratio=float(plan["band_ratio"]),
                    overlap_ratio=float(plan["overlap_ratio"]),
                )

            instance_parts: list[np.ndarray] = []
            if PER_INSTANCE_RELATION and str(plan["relation"]) != "inside":
                num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
                    (np.asarray(anchor, dtype=np.float32) > 0.2).astype(np.uint8), connectivity=8
                )
                total = float(anchor.shape[0] * anchor.shape[1])
                for cc in range(1, num_cc):
                    if float(cc_stats[cc, cv2.CC_STAT_AREA]) / total >= MIN_INSTANCE_AREA:
                        instance_parts.append(
                            np.asarray(anchor, dtype=np.float32) * (cc_labels == cc).astype(np.float32)
                        )
            if len(instance_parts) >= 2:
                support = None
                support_meta = {}
                for part in instance_parts:
                    part_support, part_meta = relation_support(part)
                    support = part_support if support is None else np.maximum(support, part_support)
                    if not support_meta:
                        support_meta = dict(part_meta)
                support_meta["relation_instances"] = len(instance_parts)
            else:
                support, support_meta = relation_support(np.asarray(anchor, dtype=np.float32))

            if ACCESSORY_COVER_REMOVED and str(plan["relation"]) != "inside":
                covered_words: list[str] = []
                for removed_word in removed_prompt_tokens(item)[:3]:
                    try:
                        removed_anchor, _ = sam.mask(image, removed_word, REMOVED_MAX_AREA)
                    except Exception:
                        continue
                    removed_area = float((removed_anchor > 0.5).mean())
                    if removed_area <= 0.0 or removed_area > REMOVED_MAX_AREA:
                        continue
                    support = np.maximum(
                        np.asarray(support, dtype=np.float32),
                        (np.asarray(removed_anchor, dtype=np.float32) > 0.5).astype(np.float32),
                    )
                    covered_words.append(removed_word)
                if covered_words:
                    support_meta["covered_removed_words"] = covered_words
            if int(plan["dilate"]) > 1:
                kernel = int(plan["dilate"])
                kernel = kernel + 1 if kernel % 2 == 0 else kernel
                support = cv2.dilate(support.astype(np.float32), np.ones((kernel, kernel), dtype=np.uint8), iterations=1)
            if int(plan["blur"]) > 1:
                kernel = int(plan["blur"])
                kernel = kernel + 1 if kernel % 2 == 0 else kernel
                support = cv2.GaussianBlur(support.astype(np.float32), (kernel, kernel), 0)
            anchor_area = float((anchor > 0.5).mean())
            support_area = float((support > 0.5).mean())
            save_mask(anchor, anchor_path)
            save_mask(support, support_path)
            meta.update(
                {
                    "key": key,
                    "family": item.get("family"),
                    "family_label": item.get("family_label"),
                    "chosen_phrase": chosen_phrase,
                    "support_plan": plan,
                    "anchor_mask": str(anchor_path),
                    "support_mask": str(support_path),
                    "anchor_area": anchor_area,
                    "support_area": support_area,
                }
            )
            meta.update(support_meta)
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("OK", idx, key, chosen_phrase, plan["relation"], f"area={support_area:.3f}", flush=True)
        except Exception as exc:
            status = "fail"
            error = repr(exc).replace(",", ";")
            print("FAIL", idx, key, error, flush=True)
        row = dict(item)
        if status == "ok":
            row["pp_local_mask"] = str(support_path.relative_to(PROJ))
            row["sam_anchor_mask"] = str(anchor_path.relative_to(PROJ))
            row["sam_support_meta"] = str(meta_path.relative_to(PROJ))
            row["sam_phrase"] = chosen_phrase
            row["sam_support_relation"] = str(plan["relation"])
            row["global_mask"] = False
        out_items.append(row)
        audit_rows.append(
            ",".join(
                [
                    str(idx),
                    key,
                    str(item.get("family", "")),
                    str(item.get("family_label", "")),
                    status,
                    chosen_phrase.replace(",", " "),
                    str(plan["relation"]),
                    f"{anchor_area:.6f}",
                    f"{support_area:.6f}",
                    str(support_path),
                    str(anchor_path),
                    error,
                ]
            )
        )
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(out_items, indent=1) + "\n", encoding="utf-8")
    AUDIT_CSV.write_text("\n".join(audit_rows) + "\n", encoding="utf-8")
    build_contact(out_items)
    print("WROTE", OUT_MANIFEST, "n", len(out_items), flush=True)
    print("CONTACT", CONTACT_SHEET, flush=True)


if __name__ == "__main__":
    main()
