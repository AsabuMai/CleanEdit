from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
OUT_CSV = EXP / "efficiency_context_2026-06-11.csv"
OUT_MD = EXP / "efficiency_context_2026-06-11.md"
OUT_AUDIT = EXP / "efficiency_context_2026-06-11_audit.json"

SEEDS = ["10", "11", "12"]
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
    "pillow_same_color_cable_knit",
    "pillow_same_color_cable_knit_grey",
    "pillow_same_color_cable_knit_armchair",
]
METHOD_SOURCES = {
    "base_only": [
        "e4_t1_t4_reconstruction_floor_metrics.csv",
        "table2_t5_internal_metrics.csv",
    ],
    "direct_target": [
        "e1_t1_t4_directgeneric_metrics.csv",
        "table2_t5_internal_metrics.csv",
    ],
    "adaptive_full_generic_support": [
        "e1_t1_t4_directgeneric_metrics.csv",
        "table2_t5_internal_metrics.csv",
    ],
    "support_v3_fixed": [
        "table2a_e4_common_subset_clipdino_metrics.csv",
        "table2_t5_internal_metrics.csv",
    ],
    "support_v3_controller_rmsgap": [
        "table2a_e4_common_subset_clipdino_metrics.csv",
        "table2_t5_internal_metrics.csv",
    ],
    "flowedit": [
        "e2_t1_t4_baseline_fixed_mask_metrics.csv",
        "table2_t5_baseline_metrics.csv",
    ],
    "flowalign": [
        "e2_t1_t4_baseline_fixed_mask_metrics.csv",
        "table2_t5_baseline_metrics.csv",
    ],
    "splitflow": [
        "e2_t1_t4_baseline_fixed_mask_metrics.csv",
        "table2_t5_baseline_metrics.csv",
    ],
    "fireflow": [
        "e2_t1_t4_baseline_fixed_mask_metrics.csv",
        "table2_t5_baseline_metrics.csv",
    ],
    "rf_solver_edit": [
        "e2_t1_t4_baseline_fixed_mask_metrics.csv",
        "table2_t5_baseline_metrics.csv",
    ],
    "reflex": [
        "e2_t1_t4_baseline_fixed_mask_metrics.csv",
        "table2_t5_baseline_metrics.csv",
    ],
}
METHOD_LABELS = {
    "base_only": "RF reconstruction",
    "direct_target": "Direct target",
    "adaptive_full_generic_support": "Generic support",
    "support_v3_fixed": "Fixed CleanEdit-SD3",
    "support_v3_controller_rmsgap": "CleanEdit-SD3",
    "flowedit": "FlowEdit-SD3",
    "flowalign": "FlowAlign-SD3",
    "splitflow": "SplitFlow-SD3",
    "fireflow": "FireFlow-FLUX/context",
    "rf_solver_edit": "RF-Solver-Edit-FLUX/context",
    "reflex": "ReFlex-FLUX/context",
}
BACKBONE = {
    "base_only": "SD3",
    "direct_target": "SD3",
    "adaptive_full_generic_support": "SD3",
    "support_v3_fixed": "SD3",
    "support_v3_controller_rmsgap": "SD3",
    "flowedit": "SD3",
    "flowalign": "SD3",
    "splitflow": "SD3",
    "fireflow": "FLUX/context",
    "rf_solver_edit": "FLUX/context",
    "reflex": "FLUX/context",
}
ORDER = list(METHOD_SOURCES)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if not any(value == key for key, value in row.items())]


def method_of(row: dict[str, str]) -> str:
    return row.get("method", "") or row.get("baseline", "")


def seed_of(row: dict[str, str]) -> str:
    return row.get("seed", "").removeprefix("seed_")


def as_float(value: object) -> float | None:
    if value in {"", "nan", "None", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def result_dir(row: dict[str, str]) -> Path | None:
    result = row.get("result_image", "")
    if result:
        path = Path(result)
        if not path.is_absolute():
            path = ROOT / path
        return path.parent
    run = row.get("run", "")
    if run:
        for prefix in [
            ROOT / "outputs" / "pretty_matrix",
            ROOT / "outputs" / "e2_t1_t4_baseline_matrix",
            ROOT / "outputs" / "e2_t5_baseline_matrix",
        ]:
            candidate = prefix / run
            if candidate.exists():
                return candidate
    return None


def load_metadata(run_dir: Path | None) -> dict[str, object]:
    if run_dir is None:
        return {}
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_command(run_dir: Path | None, metadata: dict[str, object]) -> str:
    if run_dir is not None and (run_dir / "command.txt").exists():
        return (run_dir / "command.txt").read_text(encoding="utf-8", errors="ignore")
    return str(metadata.get("command", ""))


def parse_nfe(metadata: dict[str, object], command: str) -> str:
    step_value = metadata.get("num_inference_steps")
    n_max = metadata.get("n_max")
    if step_value not in {"", None}:
        if n_max not in {"", None}:
            return f"T_steps={step_value},n_max={n_max}"
        return str(step_value)
    for key in ("nfe", "NFE"):
        value = metadata.get(key)
        if value not in {"", None}:
            return str(value)
    haystack = command + " " + str(metadata.get("matched_conditions", ""))
    patterns = [
        r"--num-inference-steps[=\s]+(\d+)",
        r"--num_steps[=\s]+(\d+)",
        r"--NFE[=\s]+(\d+)",
        r"num_inference_steps=(\d+)",
        r"num_steps=(\d+)",
        r"NFE=(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, haystack)
        if match:
            return match.group(1)
    t_steps = re.search(r"T_steps=(\d+)", haystack)
    if t_steps:
        n_max_match = re.search(r"n_max=(\d+)", haystack)
        if n_max_match:
            return f"T_steps={t_steps.group(1)},n_max={n_max_match.group(1)}"
        return f"T_steps={t_steps.group(1)}"
    return "not_recorded"


def parse_resolution(row: dict[str, str], metadata: dict[str, object]) -> str:
    resolution = metadata.get("resolution")
    if isinstance(resolution, list) and len(resolution) == 2:
        return f"{resolution[0]}x{resolution[1]}"
    result = row.get("result_image", "")
    if result:
        path = Path(result)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            with Image.open(path) as image:
                return f"{image.width}x{image.height}"
    return "not_recorded"


def bool_note(metadata: dict[str, object], command: str, needles: list[str]) -> str:
    text = (command + " " + json.dumps(metadata, sort_keys=True)).lower()
    return "yes" if any(needle in text for needle in needles) else "not_recorded"


def collect_rows(method: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename in METHOD_SOURCES[method]:
        for row in read_csv(EXP / filename):
            if method_of(row) == method and row.get("task", "") in TASKS and seed_of(row) in SEEDS:
                row["_source_file"] = filename
                rows.append(row)
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("task", ""), seed_of(row))
        if key in expected and key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def summarize_method(method: str, rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, object]]:
    enriched = []
    for row in rows:
        run_dir = result_dir(row)
        metadata = load_metadata(run_dir)
        command = load_command(run_dir, metadata)
        runtime = as_float(row.get("runtime_seconds")) or as_float(metadata.get("runtime_seconds"))
        peak = as_float(row.get("peak_gpu_memory_gb")) or as_float(metadata.get("peak_gpu_memory_gb"))
        enriched.append(
            {
                "runtime": runtime,
                "peak": peak,
                "nfe": parse_nfe(metadata, command),
                "resolution": parse_resolution(row, metadata),
                "inversion": bool_note(metadata, command, ["inversion", "invert"]),
                "offload": bool_note(metadata, command, ["--offload", "model_offload=1", "offload true"]),
            }
        )
    expected_count = len(TASKS) * len(SEEDS)
    runtime_values = [item["runtime"] for item in enriched if item["runtime"] is not None]
    peak_values = [item["peak"] for item in enriched if item["peak"] is not None]
    nfe_values = sorted({item["nfe"] for item in enriched})
    resolution_values = sorted({item["resolution"] for item in enriched})
    inversion_values = sorted({item["inversion"] for item in enriched})
    offload_values = sorted({item["offload"] for item in enriched})
    row = {
        "method": method,
        "label": METHOD_LABELS[method],
        "backbone": BACKBONE[method],
        "n": str(len(rows)),
        "expected_n": str(expected_count),
        "nfe": ",".join(nfe_values) if nfe_values else "not_recorded",
        "resolution": ",".join(resolution_values) if resolution_values else "not_recorded",
        "runtime_s_mean": f"{mean(runtime_values):.2f}" if runtime_values else "not_recorded",
        "runtime_s_available": f"{len(runtime_values)}/{len(rows)}",
        "peak_gpu_gb_mean": f"{mean(peak_values):.2f}" if peak_values else "not_recorded",
        "peak_gpu_gb_available": f"{len(peak_values)}/{len(rows)}",
        "inversion_recorded": ",".join(inversion_values) if inversion_values else "not_recorded",
        "offload_recorded": ",".join(offload_values) if offload_values else "not_recorded",
    }
    audit = {
        "method": method,
        "expected_rows": expected_count,
        "found_rows": len(rows),
        "runtime_available": len(runtime_values),
        "peak_available": len(peak_values),
        "nfe_values": nfe_values,
        "resolution_values": resolution_values,
    }
    return row, audit


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, str]]) -> str:
    fields = [
        "label",
        "backbone",
        "n",
        "nfe",
        "resolution",
        "runtime_s_mean",
        "runtime_s_available",
        "peak_gpu_gb_mean",
        "peak_gpu_gb_available",
        "offload_recorded",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines)


def main() -> int:
    rows = []
    audits = []
    for method in ORDER:
        method_rows = collect_rows(method)
        row, audit = summarize_method(method, method_rows)
        rows.append(row)
        audits.append(audit)
    row_counts_ok = all(item["found_rows"] == item["expected_rows"] for item in audits)
    nfe_ok = all(item["nfe_values"] and item["nfe_values"] != ["not_recorded"] for item in audits)
    resolution_ok = all(item["resolution_values"] and item["resolution_values"] != ["not_recorded"] for item in audits)
    audit_doc = {
        "status": "complete" if (row_counts_ok and nfe_ok and resolution_ok) else "incomplete",
        "scope": "Phase2 15-task common subset; runtime/peak memory are reported only when recorded by the producing runner.",
        "row_counts_ok": row_counts_ok,
        "nfe_ok": nfe_ok,
        "resolution_ok": resolution_ok,
        "methods": audits,
    }
    write_csv(OUT_CSV, rows)
    OUT_AUDIT.write_text(json.dumps(audit_doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Efficiency Context Table (2026-06-11)",
        "",
        f"Audit status: `{audit_doc['status']}` (`{OUT_AUDIT.relative_to(ROOT)}`).",
        "",
        "Runtime and peak GPU memory are averaged only over runs whose producer recorded those fields. Missing external-baseline runtime/peak fields are disclosed by the availability columns rather than imputed.",
        "",
        md_table(rows),
        "",
        "Interpretation: this is a cost/context table, not an efficiency claim. Native/context rows use different backbones and interfaces.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"status={audit_doc['status']}")
    print(f"row_counts_ok={row_counts_ok} nfe_ok={nfe_ok} resolution_ok={resolution_ok}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0 if audit_doc["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
