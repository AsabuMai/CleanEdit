from __future__ import annotations

import argparse
import shlex
from pathlib import Path

TASKS = [
    "cat_crown",
    "dog_bow_tie_phase2",
    "dog_front_sunglasses_phase2",
    "bowl_apple_inside",
    "white_bowl_orange_tabletop_phase2",
    "brown_bowl_lemon_phase2",
    "tshirt_star",
    "mug_heart",
    "tote_leaf",
    "red_office_chair_to_blue_office_chair",
    "green_mug_orange_phase2",
    "yellow_vase_blue_phase2",
]
SEEDS = ["10", "11", "12"]


def set_arg(tokens: list[str], flag: str, value: str) -> list[str]:
    out = list(tokens)
    if flag in out:
        idx = out.index(flag)
        out[idx + 1] = value
    else:
        out.extend([flag, value])
    return out


def remove_flag(tokens: list[str], flag: str) -> list[str]:
    return [token for token in tokens if token != flag]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare E2.4 direct_target + same fixed M_edit gating commands.")
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument("--seeds", nargs="*", default=SEEDS)
    parser.add_argument("--output-root", default="outputs/e2_support_matched_medit_gate")
    parser.add_argument("--eval-mask-dir", default="experiments/support_v3_2026-06-02/normalized_512/eval_masks")
    args = parser.parse_args()

    root = Path.cwd()
    output_root = root / args.output_root
    eval_mask_dir = root / args.eval_mask_dir
    manifest_lines: list[str] = []
    missing: list[str] = []

    for task in args.tasks:
        mask = eval_mask_dir / f"{task}_eval_mask.png"
        if not mask.exists():
            missing.append(f"missing eval mask: {mask}")
            continue
        for seed in args.seeds:
            source_cmd = root / "outputs" / "pretty_matrix" / task / "direct_target" / f"seed_{seed}" / "command.txt"
            if not source_cmd.exists():
                missing.append(f"missing source command: {source_cmd}")
                continue
            run_dir = output_root / task / "direct_target_medit_gate" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            tokens = shlex.split(source_cmd.read_text(encoding="utf-8").strip())
            tokens = set_arg(tokens, "--output", str(run_dir / "result.png"))
            tokens = set_arg(tokens, "--stats-output", str(run_dir / "stats.json"))
            tokens = set_arg(tokens, "--metadata-output", str(run_dir / "metadata.json"))
            tokens = set_arg(tokens, "--mask-output-dir", str(run_dir / "masks"))
            tokens = set_arg(tokens, "--final-edit-mask", str(mask))
            tokens = set_arg(tokens, "--final-edit-mask-mode", "replace")
            tokens = set_arg(tokens, "--object-mask-provider", "attention_velocity")
            tokens = remove_flag(tokens, "--mask-blend")
            command = " ".join(shlex.quote(token) for token in tokens)
            command_path = run_dir / "command.txt"
            command_path.write_text(command + "\n", encoding="utf-8")
            manifest_lines.append(str(command_path))

    manifest = output_root / "command_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    status = output_root / "prepare_status.txt"
    status.write_text(
        f"commands={len(manifest_lines)}\n" + "\n".join(missing) + ("\n" if missing else ""),
        encoding="utf-8",
    )
    print(f"wrote {manifest}")
    print(f"commands={len(manifest_lines)}")
    if missing:
        print(f"missing={len(missing)}")
        for item in missing:
            print(item)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
