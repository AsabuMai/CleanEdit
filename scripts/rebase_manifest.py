#!/usr/bin/env python3
"""Create a machine-local FlowEdit manifest without changing canonical evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--flowedit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    image_dir = args.flowedit_root.expanduser().resolve() / "Data" / "Images"
    missing: list[str] = []

    for entry in entries:
        source = Path(entry["image"])
        rebased = image_dir / source.name
        entry["image"] = str(rebased)
        if args.check and not rebased.is_file():
            missing.append(str(rebased))

    if missing:
        preview = "\n".join(missing[:10])
        raise SystemExit(f"missing {len(missing)} source images; first paths:\n{preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
