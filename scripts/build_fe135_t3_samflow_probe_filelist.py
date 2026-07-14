from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build tar filelist for FE135 T3 Sam-Flow probe outputs.")
    ap.add_argument("--probe-manifest", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v1_manifest.csv"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/fe135_t3_samflow_probe_v1"))
    ap.add_argument("--out", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v1_filelist.txt"))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.probe_manifest.open(newline="", encoding="utf-8")))
    paths: list[str] = [
        str(args.probe_manifest),
        "remote_patch/fe135_t3_samflow_probe_v1_sd3_rep.sbatch",
        "remote_patch/fe135_t3_samflow_probe_v1_sd3_all.sbatch",
        "remote_patch/fe135_t3_samflow_probe_v1_flux_all.sbatch",
    ]
    for row in rows:
        run_dir = args.output_root / row["baseline"] / row["task"] / f"seed_{row['seed']}"
        paths.extend(
            [
                str(run_dir / "result.png"),
                str(run_dir / "metadata.json"),
                str(run_dir / "command.txt"),
                str(run_dir / "run.log"),
            ]
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(dict.fromkeys(paths)) + "\n", encoding="utf-8", newline="\n")
    print(args.out)
    print("paths", len(dict.fromkeys(paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
