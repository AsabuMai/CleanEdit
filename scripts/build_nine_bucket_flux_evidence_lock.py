#!/usr/bin/env python3
"""Build immutable manifests and source hashes for the nine-bucket FLUX audit."""

from __future__ import annotations

import hashlib
import json
import shutil
import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flowedit_compatible_135"
OUT = DATA / "evidence_20260714"
ASSET_ROOT: Path | None = None


BUCKETS: dict[str, dict[str, Any]] = {
    "b1_inst": {
        "recipe": "instance_final_v3",
        "sources": ["manifest_bucket1_inst_v2.json"],
        "keys": [
            "fe_038_cat_and_dog_1_lego", "fe_039_cat_and_dog_2_bronze",
            "fe_192_parrots_1_top_hat", "fe_193_parrots_2_origami",
            "fe_194_parrots_3_sculpture", "fe_195_parrots2_1_crown",
            "fe_196_penguins_1_origami", "fe_197_penguins_2_sculpture",
            "fe_221_rocks_6_colorful_wooden_blocks",
            "fe_036_cake_red_blueberries_2_raspberries",
        ],
        "old_conclusion": "Plural support reaches all instances; fe_195 FLUX crown quality remains a limitation.",
    },
    "b2_sd3_text_control": {
        "recipe": "flux_canonical_text_control",
        "sources": ["manifest_bucket23_t3.json"],
        "keys": ["fe_119_gas_station_2_iccv", "fe_134_groceries_4_eccv", "fe_238_stop_arrow_1_cvpr"],
        "old_conclusion": "FLUX is a control for the SD3 blank/text limitation; this track does not repair SD3.",
    },
    "b3_flux_ghost": {
        "recipe": "best_single_pass_text_nofinal",
        "sources": ["manifest_bucket23_t3.json"],
        "keys": ["fe_119_gas_station_2_iccv", "fe_134_groceries_4_eccv", "fe_238_stop_arrow_1_cvpr"],
        "old_conclusion": "Single-pass FLUX text replacement remains unresolved: old glyph structure or incorrect spelling persists.",
        "assets": [
            "data/flowedit_compatible_135/repair_allpass_v1/masks/fe_119_gas_station_2_iccv_C5_exact_text_needs_renderer_support.png",
            "data/flowedit_compatible_135/repair_allpass_v1/masks/fe_134_groceries_4_eccv_C5_exact_text_needs_renderer_support.png",
            "data/flowedit_compatible_135/repair_allpass_v1/masks/fe_238_stop_arrow_1_cvpr_C6_short_text_needs_text_aware_refine_support.png",
        ],
    },
    "b4_flux_noed": {
        "recipe": "kind_budget_final",
        "sources": ["manifest_bucket46_flux.json"],
        "keys": [
            "fe_010_beer_glass_1_cocktail", "fe_017_boat_silhouette_1_sailboat_white_sails_red_hull",
            "fe_019_brown_owl_1_white_owl", "fe_140_horse_4_brown_horse",
            "fe_180_milk_4_whipped_cream", "fe_207_pizza_tomato_olive_1_pepperoni",
        ],
        "old_conclusion": "Simple insertions improve with budget; backlit whole-object recolors remain under-edited.",
    },
    "b5_sd3_leak_control": {
        "recipe": "flux_canonical_recolor_control",
        "sources": ["manifest_bucket5_leak.json"],
        "keys": [
            "fe_015_bikes_5_green_bicycle", "fe_016_bikes_6_yellow_bicycle",
            "fe_084_cupcake_2_red_velvet", "fe_141_iguana_1_green_lizard",
        ],
        "old_conclusion": "The halo mechanism is SD3-specific; FLUX is the unchanged control on these recolors.",
    },
    "b6_flux_bgrep": {
        "recipe": "zero_expand_outside_lock_final",
        "sources": ["manifest_bucket6_r3.json", "manifest_bucket46_flux.json"],
        "keys": [
            "fe_146_japanese_castle_1_lego_castle", "fe_172_meditation1_1_sand_sculpture",
            "fe_175_meditation1_4_golden_statue", "fe_130_gray_bird_4_golden_sculpture",
            "fe_157_kid_running_3_sculpture", "fe_266_yellow_bulldog_10_origami_lion",
            "fe_268_yellow_bulldog_12_origami_bear",
        ],
        "old_conclusion": "Zero expansion plus outside lock repairs background drift for several cases; meditation whole-subject transforms remain limited.",
    },
    "b7_half": {
        "recipe": "full_body_mask_zero_expand",
        "sources": ["manifest_bucket7_half.json"],
        "keys": ["fe_080_corgi_1_lego_bricks", "fe_081_corgi_2_wooden_sculpture"],
        "old_conclusion": "The fixed full-body multi-box mask removes the head-only support failure.",
    },
    "b8_melt_scale": {
        "recipe": "tight_recolor_mask_zero_expand",
        "sources": ["manifest_bucket8_r2.json", "manifest_bucket6_r3.json"],
        "keys": [
            "fe_027_butterflies_1_yellow", "fe_028_butterfly_1_orange_butterfly",
            "fe_128_gray_bird_2_red_bird", "fe_130_gray_bird_4_golden_sculpture",
            "fe_157_kid_running_3_sculpture",
        ],
        "old_conclusion": "Tight recolor support prevents structure melt; zero expansion controls scale on material cases.",
        "assets": [
            "data/flowedit_compatible_118/sam_support_masks_all/fe_128_gray_bird_2_red_bird_support.png",
            "data/flowedit_compatible_118/sam_support_masks_all/fe_028_butterfly_1_orange_butterfly_support.png",
            "data/flowedit_compatible_118/sam_support_masks_plural_v2/fe_027_butterflies_1_yellow_support.png",
        ],
    },
    "b9_resid": {
        "recipe": "removed_token_support",
        "sources": ["manifest_bucket9_resid.json"],
        "keys": ["fe_046_cat_crown_1_black_top_hat"],
        "old_conclusion": "Removed-token support fixes SD3 residue; FLUX old-object residue remains tied to bucket 3.",
    },
}


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_entries(names: list[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for name in reversed(names):
        raw = json.loads((DATA / name).read_text(encoding="utf-8"))
        items = raw.get("items", raw) if isinstance(raw, dict) else raw
        entries.update({item["key"]: item for item in items})
    return entries


def locked_file(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_file() and ASSET_ROOT is not None and not Path(path_value).is_absolute():
        source = (ASSET_ROOT / path_value).resolve()
        if source.is_file():
            resolved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, resolved)
    if not resolved.is_file():
        raise SystemExit(f"missing locked asset: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha(resolved),
    }


def main() -> None:
    global ASSET_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    ASSET_ROOT = args.asset_root.resolve() if args.asset_root else None
    OUT.mkdir(parents=True, exist_ok=True)
    lock: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "backend": "FLUX.1-dev",
            "host": "h100-01.gpu01.cis.k.hosei.ac.jp",
            "seed": 10,
            "num_inference_steps": 12,
            "n_max": 10,
            "final_postprocess_mode": "none",
            "comparison_protocol": "single_pass",
            "require_clean_git": True,
        },
        "buckets": {},
    }
    for bucket, spec in BUCKETS.items():
        available = load_entries(spec["sources"])
        missing = [key for key in spec["keys"] if key not in available]
        if missing:
            raise SystemExit(f"{bucket}: missing keys {missing}")
        items = [available[key] for key in spec["keys"]]
        manifest = OUT / f"manifest_{bucket}.json"
        manifest.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
        cases: dict[str, Any] = {}
        for item in items:
            image = Path(item["image"]).resolve()
            if "_baselines/src/FlowEdit/Data/Images" not in str(image):
                raise SystemExit(f"{item['key']}: not an original FlowEdit image: {image}")
            support_assets = {
                field: locked_file(str(item[field]))
                for field in ("pp_local_mask", "mask", "final_blend_mask")
                if item.get(field)
            }
            cases[item["key"]] = {
                "entry_sha256": canonical_sha(item),
                "source_image_path": str(image),
                "source_image_size_bytes": image.stat().st_size,
                "source_image_sha256": file_sha(image),
                "support_assets": support_assets,
            }
        lock["buckets"][bucket] = {
            "recipe": spec["recipe"],
            "old_conclusion": spec["old_conclusion"],
            "manifest": str(manifest.relative_to(ROOT)),
            "manifest_sha256": file_sha(manifest),
            "expected_count": len(items),
            "recipe_assets": {
                value: locked_file(value) for value in spec.get("assets", [])
            },
            "cases": cases,
        }
    lock_path = OUT / "nine_bucket_flux_evidence.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {sum(v['expected_count'] for v in lock['buckets'].values())} bucket-case runs")
    print(lock_path)


if __name__ == "__main__":
    main()
