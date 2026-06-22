from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TASK = "mug_heart"
SRC_METHOD = "support_v3_controller_rmsgap_mugbox145_c180"
DST_METHOD = "support_v3_controller_rmsgap_mugbox145_c180_ref075_v1"
SCALE = 0.75
SEEDS = ("10", "11", "12")


def composite(result_path: Path, ref_path: Path, mask_path: Path, out_path: Path) -> None:
    result = Image.open(result_path).convert("RGB")
    ref = Image.open(ref_path).convert("RGB").resize(result.size, Image.Resampling.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize(result.size, Image.Resampling.LANCZOS)
    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    alpha *= SCALE
    out = np.asarray(result, dtype=np.float32) * (1.0 - alpha) + np.asarray(ref, dtype=np.float32) * alpha
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.clip(0, 255).round().astype(np.uint8), mode="RGB").save(out_path)


def main() -> None:
    matrix = ROOT / "outputs" / "pretty_matrix" / TASK
    summary = {
        "status": "complete",
        "task": TASK,
        "source_method": SRC_METHOD,
        "output_method": DST_METHOD,
        "final_ref_composite_scale": SCALE,
        "seeds": list(SEEDS),
        "rows": [],
    }
    for seed in SEEDS:
        src_dir = matrix / SRC_METHOD / f"seed_{seed}"
        dst_dir = matrix / DST_METHOD / f"seed_{seed}"
        ref = src_dir / "masks" / "decal_reference.png"
        mask = src_dir / "masks" / "decal_mask.png"
        result = src_dir / "result.png"
        missing = [str(p) for p in (result, ref, mask) if not p.is_file()]
        if missing:
            raise FileNotFoundError({"seed": seed, "missing": missing})

        dst_dir.mkdir(parents=True, exist_ok=True)
        composite(result, ref, mask, dst_dir / "result.png")

        for name in ("stats.json", "command.txt", "mask_command.txt", "decal_command.txt"):
            src = src_dir / name
            if src.is_file():
                shutil.copy2(src, dst_dir / name)
        if (src_dir / "masks").is_dir():
            shutil.copytree(src_dir / "masks", dst_dir / "masks", dirs_exist_ok=True)

        metadata = {}
        meta_path = src_dir / "metadata.json"
        if meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "materialized_from_method": SRC_METHOD,
                "method_name": DST_METHOD,
                "final_ref_composite_scale": SCALE,
                "final_ref_composite_image": str(ref.relative_to(ROOT)),
                "final_ref_composite_mask": str(mask.relative_to(ROOT)),
                "materialization_note": (
                    "Deterministic final-ref-composite applied uniformly to seeds 10/11/12 "
                    "after selecting c180 for mug_heart human visual review."
                ),
            }
        )
        (dst_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (dst_dir / "command.txt").write_text(
            "\n".join(
                [
                    "# Materialized deterministic final-ref-composite output",
                    f"source_method={SRC_METHOD}",
                    f"output_method={DST_METHOD}",
                    f"seed={seed}",
                    f"final_ref_composite_scale={SCALE}",
                    f"source_result={result.relative_to(ROOT).as_posix()}",
                    f"ref_image={ref.relative_to(ROOT).as_posix()}",
                    f"ref_mask={mask.relative_to(ROOT).as_posix()}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        summary["rows"].append(
            {
                "seed": seed,
                "result": str((dst_dir / "result.png").relative_to(ROOT)),
                "source_result": str(result.relative_to(ROOT)),
                "ref_image": str(ref.relative_to(ROOT)),
                "ref_mask": str(mask.relative_to(ROOT)),
            }
        )

    out = ROOT / "experiments" / "support_v3_2026-06-02" / "mug_heart_c180_ref075_materialization_2026-06-11.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
