from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    metadata_paths = sorted(path for root in args.roots for path in root.glob("**/seed_10/metadata.json"))
    errors: list[str] = []
    if len(metadata_paths) != args.expected:
        errors.append(f"metadata count {len(metadata_paths)} != expected {args.expected}")

    required = {
        "run_config_sha256",
        "git_commit",
        "git_dirty",
        "evaluation_eligible",
        "eligibility_reasons",
        "postprocess_applied",
    }
    for path in metadata_paths:
        data = json.loads(path.read_text())
        missing = sorted(required - data.keys())
        checks = {
            "seed": data.get("seed") == 10,
            "steps": data.get("num_inference_steps") == 12,
            "n_max": data.get("n_max") == 10,
            "purpose": data.get("run_purpose") == "evaluation",
            "protocol": data.get("comparison_protocol") == "single_pass",
            "mode": data.get("final_postprocess_mode") == "none",
            "mask_blend": float(data.get("final_mask_blend_scale", -1)) == 0.0,
            "recolor_blend": float(data.get("final_recolor_blend_scale", -1)) == 0.0,
            "postprocess": data.get("postprocess_applied") is False,
            "eligible": data.get("evaluation_eligible") is True,
            "hash": len(str(data.get("run_config_sha256", ""))) == 64,
            "result": (path.parent / "result.png").is_file(),
        }
        if missing or not all(checks.values()):
            failed = [name for name, ok in checks.items() if not ok]
            errors.append(f"{path}: missing={missing} failed={failed}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"verified {len(metadata_paths)} eligible no-final FLUX runs")


if __name__ == "__main__":
    main()
