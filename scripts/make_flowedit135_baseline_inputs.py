import csv
import json
import os
from pathlib import Path

import yaml


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
DATA_DIR = PROJ / "data/flowedit_compatible_135"
MANIFEST = DATA_DIR / "manifest.json"
INPUT_DIR = DATA_DIR / "unique_inputs"

FLOWEDIT_DATASET = PROJ / "_baselines/src/FlowEdit/Data/flowedit135_dataset.yaml"
FLOWEDIT_FLUX_EXP = PROJ / "_baselines/src/FlowEdit/flowedit135_flux_exp.yaml"
FLOWEDIT_SD3_EXP = PROJ / "_baselines/src/FlowEdit/flowedit135_sd3_exp.yaml"
SPLITFLOW_DATASET = PROJ / "_baselines/src/SplitFlow/flowedit135_dataset.yaml"
SPLITFLOW_EXP = PROJ / "_baselines/src/SplitFlow/flowedit135_sd3_exp.yaml"

CSV_FIELDS = [
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


def unique_image(item: dict) -> Path:
    src = Path(item["image"])
    dst = INPUT_DIR / f"{item['key']}{src.suffix.lower() or '.png'}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
        except OSError:
            from shutil import copy2

            copy2(src, dst)
    return dst


def csv_rows(manifest: list[dict], baseline: str) -> list[dict]:
    rows = []
    for item in manifest:
        rows.append(
            {
                "baseline": baseline,
                "task": item["key"],
                "seed": "10",
                "status": "",
                "source_image": str(unique_image(item)),
                "source_prompt": item["source_prompt"],
                "target_prompt": item["target_prompt"],
                "result_image": "",
                "metadata": "",
                "command": "",
                "matched_conditions": "",
                "failure_reason": "",
                "notes": "FlowEdit-compatible 135 main benchmark",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_flowedit_yaml(manifest: list[dict]) -> None:
    dataset = []
    split_dataset = []
    for item in manifest:
        image = str(unique_image(item))
        dataset.append(
            {
                "input_img": image,
                "source_prompt": item["source_prompt"],
                "target_prompts": [item["target_prompt"]],
                "target_codes": [item["key"]],
            }
        )
        split_dataset.append(
            {
                "input_img": image,
                "source_prompt": [item["source_prompt"]],
                "target_prompts": [item["target_prompt"]],
                "target_codes": [item["key"]],
            }
        )
    yaml.safe_dump(dataset, FLOWEDIT_DATASET.open("w"), sort_keys=False, allow_unicode=True)
    yaml.safe_dump(split_dataset, SPLITFLOW_DATASET.open("w"), sort_keys=False, allow_unicode=True)
    yaml.safe_dump(
        [
            {
                "exp_name": "FlowEdit_FE135_FLUX",
                "dataset_yaml": "Data/flowedit135_dataset.yaml",
                "model_type": "FLUX",
                "sampler_type": "FlowEditFLUX",
                "T_steps": 28,
                "n_avg": 1,
                "src_guidance_scale": 1.5,
                "tar_guidance_scale": 5.5,
                "n_min": 0,
                "n_max": 24,
                "seed": 10,
            }
        ],
        FLOWEDIT_FLUX_EXP.open("w"),
        sort_keys=False,
        allow_unicode=True,
    )
    yaml.safe_dump(
        [
            {
                "exp_name": "FlowEdit_FE135_SD3",
                "dataset_yaml": "Data/flowedit135_dataset.yaml",
                "model_type": "SD3",
                "sampler_type": "FlowEditSD3",
                "T_steps": 50,
                "n_avg": 1,
                "src_guidance_scale": 3.5,
                "tar_guidance_scale": 13.5,
                "n_min": 0,
                "n_max": 33,
                "seed": 10,
            }
        ],
        FLOWEDIT_SD3_EXP.open("w"),
        sort_keys=False,
        allow_unicode=True,
    )
    yaml.safe_dump(
        [
            {
                "exp_name": "SplitFlow_FE135_SD3",
                "dataset_yaml": "flowedit135_dataset.yaml",
                "model_type": "SD3",
                "sampler_type": "FlowEditSD3",
                "T_steps": 50,
                "n_avg": 1,
                "src_guidance_scale": 3.5,
                "tar_guidance_scale": 13.5,
                "n_min": 0,
                "n_max": 33,
                "seed": 10,
            }
        ],
        SPLITFLOW_EXP.open("w"),
        sort_keys=False,
        allow_unicode=True,
    )


def main() -> None:
    manifest = json.load(MANIFEST.open(encoding="utf-8"))
    missing = [item["key"] for item in manifest if not Path(item["image"]).exists()]
    if missing:
        raise SystemExit(f"missing source images: {missing}")
    write_csv(DATA_DIR / "baseline_fireflow.csv", csv_rows(manifest, "fireflow"))
    write_csv(DATA_DIR / "baseline_rf_solver_edit.csv", csv_rows(manifest, "rf_solver_edit"))
    write_flowedit_yaml(manifest)
    print("n", len(manifest))
    print(DATA_DIR / "baseline_fireflow.csv")
    print(DATA_DIR / "baseline_rf_solver_edit.csv")
    print(FLOWEDIT_FLUX_EXP)
    print(FLOWEDIT_SD3_EXP)
    print(SPLITFLOW_EXP)


if __name__ == "__main__":
    main()
