from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


_ARTIFACT_PATH_KEYS = {
    "output",
    "metadata_output",
    "stats_output",
    "mask_output_dir",
    "clean_diagnostics_output_dir",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def effective_config(args: Any, *, include_artifact_paths: bool = False) -> dict[str, Any]:
    config = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if include_artifact_paths or key not in _ARTIFACT_PATH_KEYS
    }
    return dict(sorted(config.items()))


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(repo_root: Path, *args: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.returncode, completed.stdout.strip()


def git_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit_rc, commit = _git(root, "rev-parse", "HEAD")
    tracked_rc, _ = _git(root, "diff-index", "--quiet", "HEAD", "--")
    untracked_rc, untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    untracked_dirty = untracked_rc == 0 and bool(untracked)
    tracked_dirty = tracked_rc != 0
    return {
        "git_commit": commit if commit_rc == 0 else None,
        "git_dirty": bool(tracked_dirty or untracked_dirty),
        "git_dirty_tracked": bool(tracked_dirty),
        "git_dirty_untracked": bool(untracked_dirty),
    }


def flux_run_provenance(args: Any, repo_root: str | Path) -> dict[str, Any]:
    config = effective_config(args)
    git = git_state(repo_root)
    purpose = str(getattr(args, "run_purpose", "diagnostic"))
    protocol = str(getattr(args, "comparison_protocol", "single_pass"))
    reasons: list[str] = []
    if str(args.final_postprocess_mode) != "none":
        reasons.append("final_postprocess_mode_not_none")
    if float(args.final_mask_blend_scale) != 0.0:
        reasons.append("final_mask_blend_scale_nonzero")
    if float(args.final_recolor_blend_scale) != 0.0:
        reasons.append("final_recolor_blend_scale_nonzero")
    if float(args.final_knit_texture_scale) != 0.0:
        reasons.append("final_knit_texture_scale_nonzero")
    if (args.final_object_overlay or "").strip():
        reasons.append("final_object_overlay_configured")
    if protocol != "single_pass":
        reasons.append("comparison_protocol_not_single_pass")

    requested = purpose in {"evaluation", "paper"}
    eligible = requested and not reasons
    postprocess_applied = bool(
        args.method in {"dece_rf_flux", "dece_rf_flux_minimal"}
        and args.semantic_base_mask
        and str(args.final_postprocess_mode) != "none"
        and (
            float(args.final_mask_blend_scale) > 0.0
            or float(args.final_recolor_blend_scale) > 0.0
            or float(args.final_knit_texture_scale) > 0.0
            or bool((args.final_object_overlay or "").strip())
        )
    )
    return {
        "run_purpose": purpose,
        "comparison_protocol": protocol,
        "run_config_sha256": config_sha256(config),
        "run_config_hash_scope": "effective_args_excluding_artifact_paths",
        "run_config": config,
        **git,
        "postprocess_applied": postprocess_applied,
        "evaluation_eligible_requested": requested,
        "evaluation_eligible": eligible,
        "eligibility_reasons": reasons,
        "paper_use": bool(purpose == "paper" and eligible),
        "diagnostic_only": not eligible,
    }
