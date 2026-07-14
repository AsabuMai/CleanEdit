from __future__ import annotations

import json
import os

try:
    from .flux_hrec import HRecFluxEdit, build_parser, load_flux_pipeline, run_flux_edit
except ImportError:  # pragma: no cover - supports direct script execution
    from flux_hrec import HRecFluxEdit, build_parser, load_flux_pipeline, run_flux_edit  # type: ignore


def _ensure_parent(path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    result = HRecFluxEdit(args)
    _ensure_parent(args.output)
    result.images[0].save(args.output)
    metadata_output = args.metadata_output
    if metadata_output is None:
        root, _ = os.path.splitext(args.output)
        metadata_output = f"{root}_metadata.json"
    stats_output = args.stats_output
    if stats_output is None:
        root, _ = os.path.splitext(args.output)
        stats_output = f"{root}_stats.json"
    result.metadata["metadata_output"] = metadata_output
    result.metadata["stats_output"] = stats_output
    _ensure_parent(metadata_output)
    _ensure_parent(stats_output)
    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(result.metadata, f, indent=2)
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(result.stats, f, indent=2)


if __name__ == "__main__":
    main()
