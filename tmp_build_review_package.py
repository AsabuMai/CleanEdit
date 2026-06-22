import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = ROOT / "data/flowedit_compatible_135/manifest_sam_135.json"
STAGE = ROOT / "tmp_full_method_review_135_thumbs"
THUMB = 256

METHODS = [
    ("source", "source", "complete"),
    ("ours_sd3", "Ours-SD3", "complete"),
    ("ours_flux", "Ours-FLUX", "complete"),
    ("sam_flow_sd3", "SAM-Flow-SD3", "complete"),
    ("sam_flow_flux", "SAM-Flow-FLUX", "complete"),
    ("reflex", "ReFLEx", "complete"),
    ("instruct_pix2pix", "InstructPix2Pix", "complete"),
    ("ledits_pp", "LEDITS++", "complete"),
    ("fireflow", "FireFlow", "complete"),
    ("rf_solver_edit", "RF-Solver", "complete"),
    ("otrf", "OT-RF", "complete"),
    ("drfs", "DRFS", "complete"),
    ("flowedit", "FlowEdit", "partial"),
    ("flowalign", "FlowAlign", "partial"),
    ("splitflow", "SplitFlow", "partial"),
]


def resolve(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def first_glob(pattern):
    hits = sorted(ROOT.glob(pattern))
    return hits[0] if hits else None


def first_existing(paths):
    for path in paths:
        if path and path.exists():
            return path
    return None


def result_path(method, key, entry):
    if method == "source":
        return resolve(entry.get("image"))
    if method == "ours_sd3":
        return first_glob(f"outputs/fe135_subjectpreserve_full_v10_sd3/{key}/*/seed_10/result.png")
    if method == "ours_flux":
        return first_glob(f"outputs/fe135_full_dece_flux_v11c_pcie8_h100/{key}/*/seed_10/result.png")
    if method in {"sam_flow_sd3", "sam_flow_flux", "reflex"}:
        return first_glob(f"outputs/flowedit135_baselines/{method}/{key}/*/seed_10/result.png")
    if method in {"instruct_pix2pix", "ledits_pp"}:
        return first_glob(f"outputs/flowedit135_traditional_baselines/{method}/{key}/*/seed_10/result.png")
    if method in {"fireflow", "rf_solver_edit", "flowedit", "flowalign", "splitflow"}:
        return first_existing(
            [
                first_glob(f"outputs/baselines/{method}/{key}/seed_10/result.png"),
                first_glob(f"outputs/baselines/{method}/{key}/*/seed_10/result.png"),
            ]
        )
    if method == "otrf":
        return first_glob(f"_baselines/src/OT-RF/outputs/OTRF_SD3_FE135_ENH/SD3/src_{key}/tar_0/enhanced/*.png")
    if method == "drfs":
        return first_glob(f"_baselines/src/DeltaRectifiedFlowSampling/outputs/DRFS_SD3_FE135/SD3/src_{key}/tgt_0/*.png")
    return None


def load_result_image(src_path, method):
    image = Image.open(src_path).convert("RGB")
    if method == "drfs" and image.width >= image.height * 2:
        image = image.crop((image.width // 2, 0, image.width, image.height))
    return image


def save_thumbnail(src_path, out_path, method):
    image = load_result_image(src_path, method)
    image.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (THUMB, THUMB), "white")
    canvas.paste(image, ((THUMB - image.width) // 2, (THUMB - image.height) // 2))
    canvas.save(out_path, quality=90)


def mask_for(entry, size):
    mask_path = resolve(entry.get("pp_local_mask") or entry.get("mask"))
    if not mask_path or not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(mask).astype(np.float32) / 255.0 > 0.5


def image_metrics(src_path, res_path, mask, method):
    try:
        src = Image.open(src_path).convert("RGB")
        res = load_result_image(res_path, method).resize(src.size, Image.Resampling.BILINEAR)
        src_arr = np.asarray(src).astype(np.float32) / 255.0
        res_arr = np.asarray(res).astype(np.float32) / 255.0
        diff = np.abs(src_arr - res_arr).mean(axis=2)
        full_mae = float(diff.mean())
        if mask is not None and mask.any() and (~mask).any():
            inside_mae = float(diff[mask].mean())
            outside_mae = float(diff[~mask].mean())
            mask_area = float(mask.mean())
        else:
            inside_mae = full_mae
            outside_mae = full_mae
            mask_area = float(mask.mean()) if mask is not None else -1.0
        luma = res_arr.mean(axis=2)
        black_ratio = float((luma < 0.025).mean())
        white_ratio = float((luma > 0.975).mean())
        flat_gray_ratio = float(((res_arr.std(axis=2) < 0.015) & (luma > 0.15) & (luma < 0.85)).mean())
        return full_mae, inside_mae, outside_mae, mask_area, black_ratio, white_ratio, flat_gray_ratio, ""
    except Exception as exc:
        return "", "", "", "", "", "", "", repr(exc)


def bg_score(outside_mae):
    if outside_mae == "":
        return ""
    if outside_mae < 0.035:
        return 2
    if outside_mae < 0.085:
        return 1
    return 0


def naturalness_score(black_ratio, flat_gray_ratio):
    if black_ratio == "":
        return ""
    if black_ratio > 0.22 or flat_gray_ratio > 0.35:
        return 0
    if black_ratio > 0.10 or flat_gray_ratio > 0.20:
        return 1
    return 2


def auto_verdict(bg, nat):
    if bg == "" or nat == "":
        return "missing"
    if bg == 0 or nat == 0:
        return "bad"
    return "usable"


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "thumbs").mkdir(parents=True)

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    index_rows = []
    matrix_rows = []

    for entry in manifest:
        key = entry["key"]
        family = entry.get("family_label") or entry.get("family") or ""
        task_dir = STAGE / "thumbs" / key
        task_dir.mkdir(parents=True, exist_ok=True)
        src_path = result_path("source", key, entry)
        src_size = Image.open(src_path).size if src_path and src_path.exists() else None
        mask = mask_for(entry, src_size) if src_size else None

        index_entry = {
            field: entry.get(field, "")
            for field in ["key", "family_label", "config", "source_prompt", "target_prompt"]
        }
        index_entry["mask_available"] = mask is not None
        index_entry["methods"] = {}

        for method, label, coverage in METHODS:
            path = result_path(method, key, entry)
            status = "ok" if path and path.exists() else "missing"
            thumb_rel = ""
            if status == "ok":
                thumb_path = task_dir / f"{method}.jpg"
                save_thumbnail(path, thumb_path, method)
                thumb_rel = thumb_path.relative_to(STAGE).as_posix()

            metrics = ("", "", "", "", "", "", "", "")
            if status == "ok" and method != "source" and src_path and src_path.exists():
                metrics = image_metrics(src_path, path, mask, method)
            full, inside, outside, area, black, white, flat, error = metrics
            bg = bg_score(outside)
            nat = naturalness_score(black, flat)

            index_entry["methods"][method] = {
                "label": label,
                "coverage": coverage,
                "status": status,
                "thumb": thumb_rel,
                "remote_path": str(path) if path else "",
            }
            matrix_rows.append(
                {
                    "key": key,
                    "family_label": family,
                    "config": entry.get("config", ""),
                    "method": method,
                    "method_label": label,
                    "coverage": coverage,
                    "status": status,
                    "remote_path": str(path) if path else "",
                    "thumb": thumb_rel,
                    "full_mae": full,
                    "inside_mae": inside,
                    "outside_mae": outside,
                    "mask_area": area,
                    "black_ratio": black,
                    "white_ratio": white,
                    "flat_gray_ratio": flat,
                    "bg_preserve": bg,
                    "naturalness": nat,
                    "edit_success": "",
                    "subject_integrity": "",
                    "verdict": auto_verdict(bg, nat),
                    "issue_tags": "",
                    "manual_note": "" if status == "ok" else "MISSING result",
                    "metric_error": error,
                }
            )
        index_rows.append(index_entry)

    (STAGE / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (STAGE / "methods.json").write_text(
        json.dumps(
            [{"id": method, "label": label, "coverage": coverage} for method, label, coverage in METHODS],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with open(STAGE / "review_matrix_auto.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)

    print(f"tasks={len(index_rows)} rows={len(matrix_rows)} stage={STAGE}")


if __name__ == "__main__":
    main()
