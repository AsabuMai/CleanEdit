from __future__ import annotations

import csv
import json
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
SRC = PROJ / "data/flowedit_compatible_135/manifest_sweep_subset.json"
OUT_DIR = PROJ / "data/flowedit_compatible_135"
EXP = "pareto_sweep_20260701"
SEED = "10"


def unique_input(key: str) -> str:
    return str(PROJ / f"data/flowedit_compatible_135/unique_inputs/{key}.png")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def flowedit_item(item: dict) -> dict:
    key = item["key"]
    return {
        "input_img": unique_input(key),
        "source_prompt": item["source_prompt"],
        "target_prompts": [item["target_prompt"]],
        "target_codes": [key],
    }


def splitflow_item(item: dict) -> dict:
    key = item["key"]
    return {
        "input_img": unique_input(key),
        "source_prompt": [item["source_prompt"]],
        "target_prompts": [item["target_prompt"]],
        "target_codes": [key],
    }


def exp_config(exp_name: str, dataset_yaml: str, model_type: str, tar_guidance: float) -> dict:
    if model_type == "FLUX":
        return {
            "exp_name": exp_name,
            "dataset_yaml": dataset_yaml,
            "model_type": "FLUX",
            "sampler_type": "FlowEditFLUX",
            "T_steps": 28,
            "n_avg": 1,
            "src_guidance_scale": 1.5,
            "tar_guidance_scale": round(tar_guidance, 4),
            "n_min": 0,
            "n_max": 24,
            "seed": int(SEED),
        }
    return {
        "exp_name": exp_name,
        "dataset_yaml": dataset_yaml,
        "model_type": "SD3",
        "sampler_type": "FlowEditSD3",
        "T_steps": 50,
        "n_avg": 1,
        "src_guidance_scale": 3.5,
        "tar_guidance_scale": round(tar_guidance, 4),
        "n_min": 0,
        "n_max": 33,
        "seed": int(SEED),
    }


def otrf_config(exp_name: str, tar_guidance: float) -> dict:
    cfg = exp_config(exp_name, "flowedit135_pareto_non_t4.json", "SD3", tar_guidance)
    cfg.update(
        {
            "use_optimal_transport": True,
            "ot_reg_coeff": 0.1,
            "adaptive_transport": True,
            "transport_strength": 0.6,
        }
    )
    return cfg


def main() -> None:
    items = json.loads(SRC.read_text(encoding="utf-8"))
    items = [x for x in items if x.get("family_label") != "T4_local_recolor"]
    if not (25 <= len(items) <= 40):
        raise SystemExit(f"unexpected subset size: {len(items)}")

    write_json(OUT_DIR / "manifest_pareto_non_t4_33.json", items)
    (OUT_DIR / "pareto_non_t4_tasks.txt").write_text(
        "\n".join(x["key"] for x in items) + "\n", encoding="utf-8"
    )

    flowedit_data = [flowedit_item(x) for x in items]
    splitflow_data = [splitflow_item(x) for x in items]
    write_json(PROJ / "_baselines/src/FlowEdit/Data/flowedit135_pareto_non_t4.yaml", flowedit_data)
    write_json(PROJ / "_baselines/src/SplitFlow/flowedit135_pareto_non_t4.yaml", splitflow_data)
    write_json(PROJ / "_baselines/src/OT-RF/flowedit135_pareto_non_t4.json", splitflow_data)

    fields = [
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
    with (OUT_DIR / "baseline_pareto_fireflow.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "baseline": "fireflow",
                    "task": item["key"],
                    "seed": SEED,
                    "status": "",
                    "source_image": unique_input(item["key"]),
                    "source_prompt": item["source_prompt"],
                    "target_prompt": item["target_prompt"],
                    "result_image": "",
                    "metadata": "",
                    "command": "",
                    "matched_conditions": "",
                    "failure_reason": "",
                    "notes": EXP,
                }
            )

    settings = {"g070": 0.70, "g085": 0.85, "g100": 1.00, "g115": 1.15}
    fe_flux = [
        exp_config(f"../../../../outputs/{EXP}/flowedit_flux_{tag}", "Data/flowedit135_pareto_non_t4.yaml", "FLUX", 5.5 * factor)
        for tag, factor in settings.items()
    ]
    fe_sd3 = [
        exp_config(f"../../../../outputs/{EXP}/flowedit_sd3_{tag}", "Data/flowedit135_pareto_non_t4.yaml", "SD3", 13.5 * factor)
        for tag, factor in settings.items()
    ]
    sf_sd3 = [
        exp_config(f"../../../../outputs/{EXP}/splitflow_sd3_{tag}", "flowedit135_pareto_non_t4.yaml", "SD3", 13.5 * factor)
        for tag, factor in settings.items()
    ]
    ot_sd3 = [
        otrf_config(f"../../../../outputs/{EXP}/otrf_sd3_{tag}", 13.5 * factor)
        for tag, factor in settings.items()
    ]
    write_json(PROJ / "_baselines/src/FlowEdit/pareto_flux_exp.yaml", fe_flux)
    write_json(PROJ / "_baselines/src/FlowEdit/pareto_sd3_exp.yaml", fe_sd3)
    write_json(PROJ / "_baselines/src/SplitFlow/pareto_sd3_exp.yaml", sf_sd3)
    write_json(PROJ / "_baselines/src/OT-RF/pareto_sd3_enhanced_exp.yaml", ot_sd3)

    print("pareto inputs ready", len(items))


if __name__ == "__main__":
    main()
