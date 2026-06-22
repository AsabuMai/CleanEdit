#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from types import ModuleType


FIELDS = [
    "baseline",
    "task",
    "seed",
    "status",
    "source_image",
    "source_prompt",
    "target_prompt",
    "result_image",
    "metadata",
    "command",
    "matched_conditions",
    "failure_reason",
    "notes",
]


def import_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def command_text(device: str, cmd: list[str]) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={device} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        + " ".join(subprocess.list2cmdline([part]) for part in cmd)
    )


class CachedFluxEdit:
    def __init__(self, edit_py: Path, module_name: str, root_for_imports: Path) -> None:
        self.edit_py = edit_py
        self.root_for_imports = root_for_imports
        sys.path.insert(0, str(root_for_imports))
        self.module = import_module(edit_py, module_name)
        self.loaded = False
        self.t5 = None
        self.clip = None
        self.model = None
        self.ae = None
        self.name = None
        self.offload = None

    def load_once(self, args: Namespace) -> None:
        if self.loaded:
            if args.name != self.name or bool(args.offload) != bool(self.offload):
                raise RuntimeError("persistent runner requires one model/offload setting per process")
            return
        import torch

        name = args.name
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name not in self.module.configs:
            raise ValueError(f"unknown model name: {name}")
        self.t5 = self.module.load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
        self.clip = self.module.load_clip(torch_device)
        self.model = self.module.load_flow_model(name, device="cpu" if args.offload else torch_device)
        self.ae = self.module.load_ae(name, device="cpu" if args.offload else torch_device)
        self.module.load_t5 = lambda *a, **k: self.t5
        self.module.load_clip = lambda *a, **k: self.clip
        self.module.load_flow_model = lambda *a, **k: self.model
        self.module.load_ae = lambda *a, **k: self.ae
        self.loaded = True
        self.name = name
        self.offload = bool(args.offload)

    def run(self, args: Namespace, log_path: Path) -> None:
        self.load_once(args)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        old_cwd = Path.cwd()
        with log_path.open("a", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                os.chdir(self.root_for_imports)
                try:
                    self.module.main(args, device="cuda")
                finally:
                    os.chdir(old_cwd)


def fail_row(row: dict[str, str], repo_root: Path, run_dir: Path, failure: str, command: str, note: str) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "baseline": row.get("baseline", ""),
                "task": row.get("task", ""),
                "seed": row.get("seed", ""),
                "source_image": row.get("source_image", ""),
                "source_prompt": row.get("source_prompt", ""),
                "target_prompt": row.get("target_prompt", ""),
                "failure_reason": failure,
                "command": command,
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
            "command": str((run_dir / "command.txt").relative_to(repo_root)) if (run_dir / "command.txt").exists() else "",
            "failure_reason": failure,
            "notes": note,
        }
    )
    return row


def run_fireflow_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    helper: ModuleType,
    session: CachedFluxEdit,
    python_bin: str,
    device: str,
    max_image_size: int,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, str]:
    task = row["task"]
    seed = str(row["seed"]).removeprefix("seed_")
    run_dir = repo_root / "outputs" / "baselines" / "fireflow" / task / f"seed_{seed}"
    tmp_dir = run_dir / "tmp_fireflow"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = (repo_root / row["source_image"]).resolve()
    image_path, prepared_size = helper.prepare_input_image(source_image_path, tmp_dir / "input_512.png", max_image_size)
    prefix = f"fireflow_{task}_seed_{seed}"
    output_dir = tmp_dir / "fireflow_output"
    feature_dir = tmp_dir / "feature"
    args = Namespace(
        name="flux-dev",
        source_img_dir=str(image_path),
        source_prompt=row["source_prompt"],
        target_prompt=row["target_prompt"],
        feature_path=str(feature_dir),
        guidance=2,
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
        seed=int(seed),
    )
    cmd = [python_bin, str(session.edit_py), "--source_prompt", row["source_prompt"], "--target_prompt", row["target_prompt"], "--source_img_dir", str(image_path)]
    text = command_text(device, cmd) + "  # persistent process; full args are in metadata/run.log"
    (run_dir / "command.txt").write_text(text + "\n", encoding="utf-8")
    existing = helper.find_fireflow_result(output_dir, prefix)
    if existing.exists() and not dry_run and not overwrite:
        return helper.complete_row(row, repo_root=repo_root, run_dir=run_dir, task=task, seed=seed, source_image_path=source_image_path, image_path=image_path, prepared_size=prepared_size, fireflow_root=session.root_for_imports.parent, fireflow_result=existing, command_text=text)
    if dry_run:
        row.update({"status": "pending", "command": str((run_dir / "command.txt").relative_to(repo_root)), "notes": "dry_run; persistent FireFlow FLUX-dev runner"})
        return row
    try:
        session.run(args, run_dir / "run.log")
        result = helper.find_fireflow_result(output_dir, prefix)
        if not result.exists():
            raise FileNotFoundError(f"missing FireFlow output: {result}")
        return helper.complete_row(row, repo_root=repo_root, run_dir=run_dir, task=task, seed=seed, source_image_path=source_image_path, image_path=image_path, prepared_size=prepared_size, fireflow_root=session.root_for_imports.parent, fireflow_result=result, command_text=text)
    except Exception as exc:  # noqa: BLE001
        return fail_row(row, repo_root, run_dir, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", text, "persistent FireFlow FLUX-dev runner")


def run_rf_solver_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    helper: ModuleType,
    session: CachedFluxEdit,
    python_bin: str,
    device: str,
    max_image_size: int,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, str]:
    task = row["task"]
    seed = str(row["seed"]).removeprefix("seed_")
    run_dir = repo_root / "outputs" / "baselines" / "rf_solver_edit" / task / f"seed_{seed}"
    tmp_dir = run_dir / "tmp_rf_solver"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = (repo_root / row["source_image"]).resolve()
    image_path, prepared_size = helper.prepare_input_image(source_image_path, tmp_dir / "input_512.png", max_image_size)
    output_dir = tmp_dir / "rf_solver_output"
    feature_dir = tmp_dir / "feature"
    args = Namespace(
        name="flux-dev",
        source_img_dir=str(image_path),
        source_prompt=row["source_prompt"],
        target_prompt=row["target_prompt"],
        feature_path=str(feature_dir),
        guidance=2,
        num_steps=15,
        inject=3,
        output_dir=str(output_dir),
        offload=True,
    )
    cmd = [python_bin, str(session.edit_py), "--source_prompt", row["source_prompt"], "--target_prompt", row["target_prompt"], "--source_img_dir", str(image_path)]
    text = command_text(device, cmd) + "  # persistent process; full args are in metadata/run.log"
    (run_dir / "command.txt").write_text(text + "\n", encoding="utf-8")
    existing = helper.find_result(output_dir, "")
    if existing.exists() and not dry_run and not overwrite:
        return helper.complete_row(row, repo_root=repo_root, run_dir=run_dir, task=task, seed=seed, source_image_path=source_image_path, image_path=image_path, prepared_size=prepared_size, rf_solver_root=session.root_for_imports.parent, rf_solver_result=existing, command_text=text)
    if dry_run:
        row.update({"status": "pending", "command": str((run_dir / "command.txt").relative_to(repo_root)), "notes": "dry_run; persistent RF-Solver-Edit FLUX-dev runner"})
        return row
    try:
        session.run(args, run_dir / "run.log")
        result = helper.find_result(output_dir, "")
        if not result.exists():
            raise FileNotFoundError(f"missing RF-Solver-Edit output: {result}")
        return helper.complete_row(row, repo_root=repo_root, run_dir=run_dir, task=task, seed=seed, source_image_path=source_image_path, image_path=image_path, prepared_size=prepared_size, rf_solver_root=session.root_for_imports.parent, rf_solver_result=result, command_text=text)
    except Exception as exc:  # noqa: BLE001
        return fail_row(row, repo_root, run_dir, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", text, "persistent RF-Solver-Edit FLUX-dev runner")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FLUX baselines with one model load per baseline process.")
    parser.add_argument("--manifest", default="experiments/support_v3_2026-06-02/e2_t5_formal_baseline_manifest.csv", type=Path)
    parser.add_argument("--baselines", default="fireflow", help="Run one of: fireflow, rf_solver_edit. Use one process per baseline.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Run even if the baseline output already exists.")
    parser.add_argument("--fireflow-root", default="_baselines/src/FireFlow", type=Path)
    parser.add_argument("--rf-solver-root", default="_baselines/src/RF-Solver-Edit/FLUX_Image_Edit", type=Path)
    parser.add_argument("--fireflow-python", default="_baselines/envs/fireflow-py310/bin/python")
    parser.add_argument("--rf-solver-python", default="_baselines/envs/rf-solver-edit-py310/bin/python")
    args = parser.parse_args()

    repo_root = Path.cwd()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("DISABLE_BASELINE_NSFW", "1")
    rows = read_manifest(args.manifest)
    baseline_filter = {item for item in args.baselines.split() if item}
    if len(baseline_filter) != 1:
        raise SystemExit("Use one persistent process per baseline, e.g. --baselines fireflow or --baselines rf_solver_edit")
    task_filter = {item for item in args.tasks.split() if item}
    seed_filter = {item.removeprefix("seed_") for item in args.seeds.split() if item}

    fireflow_helper = import_module(repo_root / "scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py", "fireflow_legacy_helper")
    rf_helper = import_module(repo_root / "scripts/archive_legacy_2026-05-11/run_rf_solver_edit_baseline.py", "rf_solver_legacy_helper")
    sessions: dict[str, CachedFluxEdit] = {}
    processed = 0
    for index, row in enumerate(rows):
        baseline = row.get("baseline", "")
        if baseline not in baseline_filter or baseline not in {"fireflow", "rf_solver_edit"}:
            continue
        if task_filter and row.get("task") not in task_filter:
            continue
        if seed_filter and str(row.get("seed", "")).removeprefix("seed_") not in seed_filter:
            continue
        if args.skip_complete and row.get("status") == "complete":
            continue
        if args.limit and processed >= args.limit:
            break
        if baseline == "fireflow":
            if baseline not in sessions:
                sessions[baseline] = (
                    Namespace(edit_py=repo_root / args.fireflow_root / "src/edit.py", root_for_imports=repo_root / args.fireflow_root / "src")
                    if args.dry_run
                    else CachedFluxEdit(repo_root / args.fireflow_root / "src/edit.py", "fireflow_persistent_edit", repo_root / args.fireflow_root / "src")
                )
            rows[index] = run_fireflow_row(row, repo_root=repo_root, helper=fireflow_helper, session=sessions[baseline], python_bin=args.fireflow_python, device=args.device, max_image_size=args.max_image_size, dry_run=args.dry_run, overwrite=args.overwrite)
        elif baseline == "rf_solver_edit":
            if baseline not in sessions:
                sessions[baseline] = (
                    Namespace(edit_py=repo_root / args.rf_solver_root / "src/edit.py", root_for_imports=repo_root / args.rf_solver_root / "src")
                    if args.dry_run
                    else CachedFluxEdit(repo_root / args.rf_solver_root / "src/edit.py", "rf_solver_persistent_edit", repo_root / args.rf_solver_root / "src")
                )
            rows[index] = run_rf_solver_row(row, repo_root=repo_root, helper=rf_helper, session=sessions[baseline], python_bin=args.rf_solver_python, device=args.device, max_image_size=args.max_image_size, dry_run=args.dry_run, overwrite=args.overwrite)
        processed += 1
        write_manifest(args.manifest, rows)
        print(f"processed {processed}: {baseline} {row.get('task')} seed={row.get('seed')} status={rows[index].get('status')}", flush=True)
    write_manifest(args.manifest, rows)
    print(f"processed {processed} persistent FLUX baseline row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
