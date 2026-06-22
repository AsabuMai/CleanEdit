from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_118/manifest.json"
OUT_DIR = PROJ / "data/flowedit_compatible_118/eval_masks"
OVERLAY_DIR = OUT_DIR / "overlays"
AUDIT_CSV = OUT_DIR / "_audit.csv"
AUDIT_JSON = OUT_DIR / "_audit.json"
CONTACT = OUT_DIR / "_contact_sheet.jpg"

MODEL_NAME = os.environ.get("GROUNDING_MODEL", "IDEA-Research/grounding-dino-base")
BOX_THRESHOLD = float(os.environ.get("BOX_THRESHOLD", "0.16"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.18"))


KEY_PHRASE_FIX = {
    "fe_017_boat_silhouette_1_sailboat_white_sails_red_hull": ["sailboat", "boat"],
    "fe_024_bus_2_volkswagen_logo": ["van", "front of the van", "yellow van"],
    "fe_027_butterflies_1_yellow": ["butterflies", "butterfly"],
    "fe_084_cupcake_2_red_velvet": ["cupcake", "cream"],
    "fe_114_flowers_1_orange_yellow_white": ["flowers", "bouquet", "vase"],
    "fe_115_flowers_2_blue_purple_white": ["flowers", "bouquet", "vase"],
    "fe_221_rocks_6_colorful_wooden_blocks": ["rocks", "stack of rocks"],
}

IMAGE_STEM_PHRASE_FIX = {
    "bus": ["van", "yellow van", "front of the van"],
    "free_wifi": ["board", "black board", "sign"],
    "gas_station": ["sign", "gas station sign"],
    "groceries": ["paper", "brown paper", "grocery list"],
    "luna": ["sign", "luna sign"],
    "sign": ["billboard", "sign"],
    "stop": ["stop sign", "sign"],
    "stop_arrow": ["stop sign", "sign"],
    "stop_sticker": ["sticker", "stop sticker", "sign"],
    "this_must_be_the_place": ["neon sign", "sign"],
    "cat_and_dog": ["dog and cat", "dog", "cat"],
    "parrots": ["parrots", "parrot"],
    "parrots2": ["parrots", "parrot"],
    "penguins": ["penguins", "penguin"],
    "rocks": ["rocks", "stack of rocks"],
    "yellow_bulldog": ["dog figurine", "dog"],
}

MANUAL_BOX_BY_STEM = {
    # GroundingDINO tends to expand "paper/sign" to the full image here. The
    # actual editable region is the neon text panel.
    "this_must_be_the_place": (0.06, 0.18, 0.92, 0.75),
}


def normalize_phrase(value: str) -> str:
    return value.replace(",", " ").replace("_", " ").strip().lower()


def unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = normalize_phrase(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def phrase_candidates(item: dict) -> list[str]:
    key = item["key"]
    stem = Path(item["image"]).stem
    fam = item.get("family_label", "")
    host = normalize_phrase(item.get("host_tokens", ""))
    phrases: list[str] = []
    phrases += KEY_PHRASE_FIX.get(key, [])
    phrases += IMAGE_STEM_PHRASE_FIX.get(stem, [])

    if fam == "T1_attached_accessory":
        phrases += [host, f"{host} head", "head"]
    elif fam == "T2_container_insertion":
        phrases += [host, "container", "plate", "glass", "cake"]
    elif fam == "T3_surface_decal":
        # Avoid the generic "surface" prompt; GroundingDINO often expands it to
        # the full image, which is too weak for outside-region evaluation.
        phrases += [host, "sign", "board", "paper"]
    elif fam == "T4_local_recolor":
        phrases += [host, "object"]
    elif fam == "T5_same_color_material":
        phrases += [host, "object"]
    else:
        phrases += [host, "object"]
    return unique([p for p in phrases if p and p != "object"])


def crop_box_for_task(box: tuple[float, float, float, float], item: dict, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = box
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    fam = item.get("family_label", "")

    if fam == "T1_attached_accessory":
        # Accessory edits sit on/near the head. Detect the source subject and keep the top band.
        y1 = y0 + 0.45 * bh
        x0 -= 0.12 * bw
        x1 += 0.12 * bw
    elif fam == "T2_container_insertion":
        # Additions are usually inside/on the container; keep the central/upper region.
        y0 += 0.05 * bh
        y1 -= 0.05 * bh
        x0 -= 0.06 * bw
        x1 += 0.06 * bw
    elif fam == "T3_surface_decal":
        x0 -= 0.08 * bw
        x1 += 0.08 * bw
        y0 -= 0.08 * bh
        y1 += 0.08 * bh
    else:
        x0 -= 0.06 * bw
        x1 += 0.06 * bw
        y0 -= 0.06 * bh
        y1 += 0.06 * bh

    return (
        max(0, int(math.floor(x0))),
        max(0, int(math.floor(y0))),
        min(width, int(math.ceil(x1))),
        min(height, int(math.ceil(y1))),
    )


def soft_box_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x0, y0, x1, y1 = box
    radius = max(3, int(0.04 * max(x1 - x0, y1 - y0)))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=255)
    blur = max(1, int(0.012 * max(width, height)))
    return mask.filter(ImageFilter.GaussianBlur(blur))


def fallback_mask(size: tuple[int, int], item: dict) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = size
    fam = item.get("family_label", "")
    if fam == "T3_surface_decal":
        box = (int(0.18 * width), int(0.18 * height), int(0.82 * width), int(0.70 * height))
    elif fam == "T1_attached_accessory":
        box = (int(0.25 * width), int(0.10 * height), int(0.75 * width), int(0.45 * height))
    else:
        box = (int(0.18 * width), int(0.12 * height), int(0.82 * width), int(0.88 * height))
    return soft_box_mask(size, box), box


def manual_box_mask(size: tuple[int, int], norm_box: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = size
    x0, y0, x1, y1 = norm_box
    box = (
        max(0, int(round(x0 * width))),
        max(0, int(round(y0 * height))),
        min(width, int(round(x1 * width))),
        min(height, int(round(y1 * height))),
    )
    return soft_box_mask(size, box), box


def overlay_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (255, 80, 40, 0))
    alpha = mask.point(lambda v: int(v * 0.45))
    red.putalpha(alpha)
    return Image.alpha_composite(base, red).convert("RGB")


class GroundingBoxer:
    def __init__(self):
        import torch
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = GroundingDinoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
        self.model = GroundingDinoForObjectDetection.from_pretrained(MODEL_NAME, local_files_only=True).to(self.device)
        self.model.eval()
        self.cache: dict[tuple[str, str], tuple[float, tuple[float, float, float, float]] | None] = {}
        print("device", self.device, "model", MODEL_NAME, flush=True)

    def detect(self, image_path: Path, phrase: str) -> tuple[float, tuple[float, float, float, float]] | None:
        key = (str(image_path), phrase)
        if key in self.cache:
            return self.cache[key]
        image = Image.open(image_path).convert("RGB")
        prompt = phrase if phrase.endswith(".") else f"{phrase}."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = result.get("boxes")
        scores = result.get("scores")
        if boxes is None or len(boxes) == 0:
            self.cache[key] = None
            return None
        best = int(scores.argmax().item())
        out = (float(scores[best].item()), tuple(float(v) for v in boxes[best].tolist()))
        self.cache[key] = out
        return out


def make_contact(records: list[dict]) -> None:
    thumbs = []
    for rec in records:
        overlay = Path(rec["overlay"])
        if not overlay.exists():
            continue
        img = Image.open(overlay).convert("RGB")
        img.thumbnail((220, 170))
        canvas = Image.new("RGB", (220, 205), "white")
        canvas.paste(img, ((220 - img.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        label = f"{rec['idx']:03d} {rec['family']} {rec['phrase'][:22]}"
        draw.text((4, 174), label, fill=(0, 0, 0))
        draw.text((4, 190), rec["status"], fill=(180, 40, 20) if rec["status"] != "detected" else (20, 100, 40))
        thumbs.append(canvas)
    cols = 5
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 220, rows * 205), "white")
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * 220, (i // cols) * 205))
    sheet.save(CONTACT, quality=92)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    items = json.load(MANIFEST.open())
    boxer = GroundingBoxer()
    records = []
    for idx, item in enumerate(items):
        image_path = Path(item["image"])
        image = Image.open(image_path).convert("RGB")
        stem = image_path.stem
        phrases = phrase_candidates(item)
        if stem in MANUAL_BOX_BY_STEM:
            mask, box = manual_box_mask(image.size, MANUAL_BOX_BY_STEM[stem])
            score = 1.0
            status = "manual_box"
            best_phrase = "neon text panel"
        else:
            best = None
            best_phrase = ""
            for phrase in phrases:
                hit = boxer.detect(image_path, phrase)
                if hit is not None and (best is None or hit[0] > best[0]):
                    best = hit
                    best_phrase = phrase
            if best is None:
                mask, box = fallback_mask(image.size, item)
                score = 0.0
                status = "fallback_center"
                best_phrase = "|".join(phrases[:3]) or "fallback"
            else:
                score, raw_box = best
                box = crop_box_for_task(raw_box, item, image.size)
                mask = soft_box_mask(image.size, box)
                status = "detected"
        mask_path = OUT_DIR / f"{item['key']}_eval_mask.png"
        overlay_path = OVERLAY_DIR / f"{item['key']}_overlay.jpg"
        mask.save(mask_path)
        overlay_image(image, mask).save(overlay_path, quality=92)
        area = float((np.asarray(mask) > 64).mean())
        rec = {
            "idx": idx,
            "key": item["key"],
            "family": item.get("family", ""),
            "family_label": item.get("family_label", ""),
            "phrase": best_phrase,
            "phrases_tried": ";".join(phrases),
            "score": score,
            "status": status,
            "area": area,
            "box": ",".join(str(v) for v in box),
            "mask": str(mask_path),
            "overlay": str(overlay_path),
        }
        records.append(rec)
        print(idx + 1, len(items), rec["key"], status, best_phrase, f"score={score:.3f}", f"area={area:.3f}", flush=True)
    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    AUDIT_JSON.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    make_contact(records)
    print("masks", len(records), OUT_DIR)
    print("audit", AUDIT_CSV)
    print("contact", CONTACT)
    print("fallbacks", sum(1 for rec in records if rec["status"] != "detected"))


if __name__ == "__main__":
    main()
