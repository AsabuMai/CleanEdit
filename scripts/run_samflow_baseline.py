#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image


FIELDS = [
    "baseline",
    "task",
    "seed",
    "status",
    "source_image",
    "source_prompt",
    "target_prompt",
    "source_tokens",
    "target_tokens",
    "unchanged_tokens",
    "result_image",
    "metadata",
    "command",
    "matched_conditions",
    "failure_reason",
    "notes",
]

SAM_FLOW_MODES = {
    "sam_flow_sd3": {
        "mode": "sd3",
        "label": "Sam-Flow-SD3",
        "backbone": "stabilityai/stable-diffusion-3-medium-diffusers",
        "config": "configs/sd3.yaml",
    },
    "sam_flow_flux": {
        "mode": "flux",
        "label": "Sam-Flow-FLUX/context",
        "backbone": "black-forest-labs/FLUX.1-dev",
        "config": "configs/flux.yaml",
    },
}

# Tokens are deliberately fixed per task and recorded in metadata. For addition
# edits, the source token names the source object/surface to anchor and the
# target token names the inserted/changed visual element.
TASK_TOKENS: dict[str, dict[str, list[str]]] = {
    "cat_crown": {"source": ["cat"], "target": ["crown"]},
    "dog_bow_tie_phase2": {"source": ["dog", "neck"], "target": ["bow", "tie"]},
    "dog_front_sunglasses_phase2": {"source": ["dog", "eyes"], "target": ["sunglasses"]},
    "bowl_apple_inside": {"source": ["bowl"], "target": ["apple"]},
    "white_bowl_orange_tabletop_phase2": {"source": ["table", "bowl"], "target": ["orange"]},
    "brown_bowl_lemon_phase2": {"source": ["bowl"], "target": ["lemon"]},
    "tshirt_star": {"source": ["t-shirt"], "target": ["star"]},
    "mug_heart": {"source": ["mug"], "target": ["heart"]},
    "tote_leaf": {"source": ["tote", "bag"], "target": ["leaf"]},
    "red_office_chair_to_blue_office_chair": {"source": ["red", "chair"], "target": ["blue", "chair"]},
    "green_mug_orange_phase2": {"source": ["green", "mug"], "target": ["orange", "mug"]},
    "yellow_vase_blue_phase2": {"source": ["yellow", "vase"], "target": ["blue", "vase"]},
    "pillow_same_color_cable_knit": {"source": ["pillow"], "target": ["cable-knit", "pillow"]},
    "pillow_same_color_cable_knit_grey": {"source": ["pillow"], "target": ["cable-knit", "pillow"]},
    "pillow_same_color_cable_knit_armchair": {"source": ["pillow"], "target": ["cable-knit", "pillow"]},
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_input_image(source: Path, output: Path, max_image_size: int) -> tuple[Path, list[int]]:
    image = Image.open(source).convert("RGB")
    if max(image.size) > max_image_size:
        scale = max_image_size / float(max(image.size))
        image = image.resize(
            (max(16, int(round(image.width * scale))), max(16, int(round(image.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output, [image.width, image.height]


def token_args(tokens: dict[str, list[str]]) -> list[str]:
    args: list[str] = []
    unchanged = tokens.get("unchanged", [])
    if unchanged:
        for token in unchanged:
            args.extend(["--unchanged-token", token])
        return args
    for token in tokens.get("source", []):
        args.extend(["--source-token", token])
    for token in tokens.get("target", []):
        args.extend(["--target-token", token])
    return args


def split_manifest_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    raw = str(value).strip()
    if not raw:
        return []
    if "," in raw:
        return [item.strip().lower() for item in raw.split(",") if item.strip()]
    return [raw.lower()]


def row_tokens(row: dict[str, str]) -> dict[str, list[str]] | None:
    source = split_manifest_tokens(row.get("source_tokens"))
    target = split_manifest_tokens(row.get("target_tokens"))
    unchanged = split_manifest_tokens(row.get("unchanged_tokens"))
    if unchanged:
        return {"unchanged": unchanged}
    if source and target:
        return {"source": source, "target": target}
    return None


def append_note(row: dict[str, str], note: str) -> str:
    previous = row.get("notes", "").strip()
    if not previous:
        return note
    if note in previous:
        return previous
    return f"{previous}; {note}"


def complete_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    run_dir: Path,
    task: str,
    seed: str,
    baseline: str,
    source_image_path: Path,
    prepared_image: Path,
    prepared_size: list[int],
    samflow_root: Path,
    result_path: Path,
    command_text: str,
    tokens: dict[str, list[str]],
) -> dict[str, str]:
    shutil.copy2(result_path, run_dir / "result.png")
    spec = SAM_FLOW_MODES[baseline]
    matched_conditions = (
        f"official Sam-Flow {spec['mode']} run_image.py; source image resized to max_image_size=512; "
        "source prompt, target prompt, seed, and fixed task mask tokens match the manifest; "
        f"config={spec['config']}; backbone={spec['backbone']}"
    )
    metadata = {
        "baseline": baseline,
        "label": spec["label"],
        "task": task,
        "seed": int(seed),
        "source_image": row["source_image"],
        "source_prompt": row["source_prompt"],
        "target_prompt": row["target_prompt"],
        "source_tokens": row.get("source_tokens", ""),
        "target_tokens": row.get("target_tokens", ""),
        "unchanged_tokens": row.get("unchanged_tokens", ""),
        "resolution": prepared_size,
        "tokens": tokens,
        "matched_conditions": matched_conditions,
        "samflow_root": str(samflow_root),
        "original_source_image": str(source_image_path),
        "prepared_source_image": str(prepared_image),
        "samflow_output": str(result_path),
        "command": command_text,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    row.update(
        {
            "status": "complete",
            "result_image": str((run_dir / "result.png").relative_to(repo_root)),
            "metadata": str((run_dir / "metadata.json").relative_to(repo_root)),
            "command": str((run_dir / "command.txt").relative_to(repo_root)),
            "matched_conditions": matched_conditions,
            "failure_reason": "",
            "notes": append_note(row, f"official {spec['label']} runner"),
        }
    )
    return row


def run_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    output_root: Path,
    samflow_root: Path,
    python_bin: str,
    device: str,
    max_image_size: int,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, str]:
    baseline = row["baseline"]
    if baseline not in SAM_FLOW_MODES:
        return row
    spec = SAM_FLOW_MODES[baseline]
    if not samflow_root.is_absolute():
        samflow_root = (repo_root / samflow_root).resolve()
    task = row["task"]
    seed = str(row["seed"]).removeprefix("seed_")
    tokens = row_tokens(row) or TASK_TOKENS.get(task)
    run_root = output_root if output_root.is_absolute() else repo_root / output_root
    run_dir = run_root / baseline / task / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp_samflow"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = (repo_root / row["source_image"]).resolve()
    prepared_image, prepared_size = prepare_input_image(source_image_path, tmp_dir / "input_512.png", max_image_size)
    output_root = tmp_dir / f"results_{spec['mode']}"
    target_code = f"{task}_seed_{seed}_{baseline}"
    final_output = output_root / prepared_image.stem / target_code / f"{target_code}_output.png"
    python_path = Path(python_bin)
    if not python_path.is_absolute() and ("/" in python_bin or "\\" in python_bin):
        python_bin = str(repo_root / python_path)

    cmd = [
        python_bin,
        str(samflow_root / "scripts" / "run_image.py"),
        "--mode",
        spec["mode"],
        "--config",
        str(samflow_root / spec["config"]),
        "--image",
        str(prepared_image),
        "--source-prompt",
        row["source_prompt"],
        "--target-prompt",
        row["target_prompt"],
        "--target-code",
        target_code,
        "--output-root",
        str(output_root),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if tokens:
        cmd.extend(token_args(tokens))
    else:
        row.update(
            {
                "status": "failed",
                "failure_reason": f"No Sam-Flow token mapping for task={task}",
                "notes": append_note(row, f"official {spec['label']} runner"),
            }
        )
        return row

    env = {
        "CUDA_VISIBLE_DEVICES": device,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": os.environ.get("HF_HOME", str(repo_root / ".cache" / "huggingface")),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", str(repo_root / ".cache" / "huggingface" / "hub")),
        "HUGGINGFACE_HUB_CACHE": os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            str(repo_root / ".cache" / "huggingface" / "hub"),
        ),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        "DIFFUSERS_OFFLINE": os.environ.get("DIFFUSERS_OFFLINE", "1"),
    }
    command_text = (
        f"CUDA_VISIBLE_DEVICES={device} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 "
        + " ".join(subprocess.list2cmdline([part]) for part in cmd)
    )
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    if final_output.exists() and not overwrite and not dry_run:
        return complete_row(
            row,
            repo_root=repo_root,
            run_dir=run_dir,
            task=task,
            seed=seed,
            baseline=baseline,
            source_image_path=source_image_path,
            prepared_image=prepared_image,
            prepared_size=prepared_size,
            samflow_root=samflow_root,
            result_path=final_output,
            command_text=command_text,
            tokens=tokens,
        )
    if dry_run:
        row.update(
            {
                "status": "pending",
                "command": str((run_dir / "command.txt").relative_to(repo_root)),
                "notes": append_note(row, f"dry_run; official {spec['label']} runner"),
            }
        )
        return row

    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=samflow_root,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, **env},
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Sam-Flow exited with code {completed.returncode}")
        if not final_output.exists():
            raise FileNotFoundError(f"missing Sam-Flow output: {final_output}")
        return complete_row(
            row,
            repo_root=repo_root,
            run_dir=run_dir,
            task=task,
            seed=seed,
            baseline=baseline,
            source_image_path=source_image_path,
            prepared_image=prepared_image,
            prepared_size=prepared_size,
            samflow_root=samflow_root,
            result_path=final_output,
            command_text=command_text,
            tokens=tokens,
        )
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "baseline": baseline,
                    "task": task,
                    "seed": int(seed),
                    "source_image": row["source_image"],
                    "prepared_source_image": str(prepared_image),
                    "source_prompt": row["source_prompt"],
                    "target_prompt": row["target_prompt"],
        "tokens": tokens,
        "source_tokens": row.get("source_tokens", ""),
        "target_tokens": row.get("target_tokens", ""),
        "unchanged_tokens": row.get("unchanged_tokens", ""),
                    "failure_reason": failure,
                    "command": command_text,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        row.update(
            {
                "status": "failed",
                "metadata": str((run_dir / "metadata.json").relative_to(repo_root)),
                "command": str((run_dir / "command.txt").relative_to(repo_root)),
                "failure_reason": failure,
                "notes": append_note(row, f"official {spec['label']} runner"),
            }
        )
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched Sam-Flow SD3/FLUX baselines.")
    parser.add_argument("--manifest", default="experiments/support_v3_2026-06-02/e2_t1_t4_formal_baseline_manifest.csv", type=Path)
    parser.add_argument("--samflow-root", default="_baselines/src/Sam-Flow", type=Path)
    parser.add_argument("--python", default="_baselines/envs/sam-flow-py310/bin/python")
    parser.add_argument("--output-root", default="outputs/baselines", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--baselines", default="sam_flow_sd3 sam_flow_flux")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo_root = Path.cwd()
    rows = read_manifest(args.manifest)
    task_filter = {item for item in args.tasks.split() if item}
    seed_filter = {item.removeprefix("seed_") for item in args.seeds.split() if item}
    baseline_filter = {item for item in args.baselines.split() if item}
    selected = 0
    for index, row in enumerate(rows):
        baseline = row.get("baseline", "")
        if baseline not in baseline_filter or baseline not in SAM_FLOW_MODES:
            continue
        if task_filter and row.get("task") not in task_filter:
            continue
        if seed_filter and row.get("seed", "").removeprefix("seed_") not in seed_filter:
            continue
        if args.skip_complete and row.get("status") == "complete":
            continue
        if args.limit and selected >= args.limit:
            break
        rows[index] = run_row(
            row,
            repo_root=repo_root,
            output_root=args.output_root,
            samflow_root=args.samflow_root,
            python_bin=args.python,
            device=args.device,
            max_image_size=args.max_image_size,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        selected += 1
    write_manifest(args.manifest, rows)
    print(f"processed {selected} Sam-Flow row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
