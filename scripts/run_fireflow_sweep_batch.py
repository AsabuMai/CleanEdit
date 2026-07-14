from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
sys.path.insert(0, str(PROJ))

from scripts.run_flux_persistent_baselines import CachedFluxEdit, import_module  # noqa: E402


MANIFEST = PROJ / os.environ.get("MANIFEST_CSV", "data/flowedit_compatible_135/baseline_pareto_fireflow.csv")
OUT = PROJ / os.environ.get("OUT", "outputs/pareto_sweep_20260701/fireflow_g100")
METHOD = os.environ.get("METHOD", "fireflow")
SEED = os.environ.get("SEED", "10")
GUIDANCE = float(os.environ.get("FIREFLOW_GUIDANCE", "2.0"))
LIMIT = int(os.environ.get("LIMIT", "0"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_rows(MANIFEST)
    if LIMIT:
        rows = rows[:LIMIT]

    helper = import_module(PROJ / "scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py", "pareto_fireflow_helper")
    session = CachedFluxEdit(
        PROJ / "_baselines/src/FireFlow/src/edit.py",
        "pareto_fireflow_edit",
        PROJ / "_baselines/src/FireFlow/src",
    )

    done = 0
    for row in rows:
        task = row["task"]
        final = OUT / task / METHOD / f"seed_{SEED}" / "result.png"
        if final.exists():
            print("skip", task, flush=True)
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        run_dir = final.parent / "raw"
        tmp_dir = run_dir / "tmp_fireflow"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        source_image_path = Path(row["source_image"])
        image_path, prepared_size = helper.prepare_input_image(source_image_path, tmp_dir / "input_512.png", 512)
        prefix = f"pareto_fireflow_{task}_seed_{SEED}_g{GUIDANCE:g}"
        output_dir = tmp_dir / "fireflow_output"
        feature_dir = tmp_dir / "feature"
        args = Namespace(
            name="flux-dev",
            source_img_dir=str(image_path),
            source_prompt=row["source_prompt"],
            target_prompt=row["target_prompt"],
            feature_path=str(feature_dir),
            guidance=GUIDANCE,
            num_steps=8,
            inject=1,
            start_layer_index=0,
            end_layer_index=37,
            output_dir=str(output_dir),
            output_prefix=prefix,
            sampling_strategy="fireflow",
            offload=True,
            reuse_v=1,
            editing_strategy="replace_v",
            qkv_ratio="1.0,1.0,1.0",
            seed=int(SEED),
        )
        t0 = time.time()
        try:
            session.run(args, run_dir / "run.log")
            result = helper.find_fireflow_result(output_dir, prefix)
            if not result.exists():
                raise FileNotFoundError(result)
            shutil.copy(result, final)
            meta = {
                "method": METHOD,
                "seed": int(SEED),
                "fireflow_guidance": GUIDANCE,
                "prepared_size": prepared_size,
                "source_prompt": row["source_prompt"],
                "target_prompt": row["target_prompt"],
            }
            (final.parent / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            done += 1
            print("OK", task, "%.1fs" % (time.time() - t0), flush=True)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print("FAILED", task, repr(exc), flush=True)
    print("ALLDONE pareto-fireflow", done, "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
