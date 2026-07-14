import json
from pathlib import Path

from PIL import Image

from scripts.evaluate_paper_metrics import evaluate_run, find_run_dirs


def test_evaluate_run_uses_failure_annotations(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "cat_crown" / "full" / "seed_10"
    run_dir.mkdir(parents=True)
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (128, 128, 128)).save(source)
    Image.new("RGB", (8, 8), (130, 128, 128)).save(run_dir / "result.png")
    (run_dir / "stats.json").write_text("[]\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps({"image": str(source), "source_prompt": "source", "target_prompt": "target"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text("run\n", encoding="utf-8")

    record = evaluate_run(
        run_dir,
        outputs_dir,
        failure_annotations={
            "cat_crown/full/seed_10": {
                "failure_flag": "localization_error",
                "failure_note": "crown too high",
            }
        },
    )

    assert record["complete"] is True
    assert record["failure_flag"] == "localization_error"
    assert record["failure_note"] == "crown too high"


def test_find_run_dirs_ignores_nested_support_artifacts(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "yellow_car_blue" / "full" / "seed_10"
    support_dir = run_dir / "masks" / "vehicle_paint"
    support_dir.mkdir(parents=True)

    for directory in (run_dir, support_dir):
        Image.new("RGB", (8, 8), (130, 128, 128)).save(directory / "result.png")
        (directory / "stats.json").write_text("[]\n", encoding="utf-8")
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")
        (directory / "command.txt").write_text("run\n", encoding="utf-8")

    assert find_run_dirs(outputs_dir) == [run_dir]


def test_find_run_dirs_filters_task_method_and_seed(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    keep = outputs_dir / "cat_crown" / "full" / "seed_10"
    wrong_task = outputs_dir / "red_chair_blue" / "full" / "seed_10"
    wrong_method = outputs_dir / "cat_crown" / "full_no_rec" / "seed_10"
    wrong_seed = outputs_dir / "cat_crown" / "full" / "seed_11"

    for directory in (keep, wrong_task, wrong_method, wrong_seed):
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")

    found = find_run_dirs(
        outputs_dir,
        task_names={"cat_crown"},
        method_names={"full"},
        seeds={"10"},
    )
    assert found == [keep]
