# Reproduction guide

## Scope

The checked-in release reproduces the CleanEdit method, FlowEdit-135 experiment construction, evaluation, ablations, and published numeric summaries. It does not redistribute model weights, FlowEdit source images, external baseline repositories, or generated images.

## 1. Environment

Create `.venv` at the repository root and install `requirements.txt`. On a managed cluster, perform installation, PyTorch imports, model loading, tests, and generation only on a compute node. The login node should be limited to lightweight file, Git, queue, and job-submission operations.

Set the project root once when launching scripts from outside the repository:

```bash
export CLEANEDIT_ROOT=/path/to/CleanEdit
```

## 2. Benchmark data

The canonical manifest is `data/flowedit_compatible_135/manifest.json`. Its source-image fields record the paths used in the original experiment. Create a machine-local copy rather than editing the canonical evidence file:

```bash
python scripts/rebase_manifest.py \
  --manifest data/flowedit_compatible_135/manifest.json \
  --flowedit-root /path/to/FlowEdit \
  --output data/flowedit_compatible_135/manifest.local.json \
  --check
```

The release keeps the fixed-region audit metadata but does not duplicate the
generated mask PNGs. After rebasing the manifest, regenerate them with:

```bash
python make_flowedit135_fixed_eval_masks.py
```

This writes masks, overlays, and a fresh audit under
`data/flowedit_compatible_135/eval_masks/`.

## 3. CleanEdit generation

The two public batch drivers are:

- `scripts/pie_sd3_batch_kindaware.py`
- `scripts/pie_batch_flux_kindaware_v11.py`

Both accept `MANIFEST`, `START`, `LIMIT`, `SEED`, and an output-root environment variable. The Slurm wrappers set the fair main protocol:

```bash
export ABLATE=no_final_postprocess
```

This disables final mask blending and recolor compositing. Do not combine post-processed outputs with the main benchmark.

Typical Slurm launch:

```bash
MANIFEST=$PWD/data/flowedit_compatible_135/manifest.local.json \
OUT=outputs/flowedit135_sd3 \
sbatch slurm/run_sd3_flowedit135.sbatch

MANIFEST=$PWD/data/flowedit_compatible_135/manifest.local.json \
FLUX_OUT=outputs/flowedit135_flux \
sbatch slurm/run_flux_flowedit135.sbatch
```

Use `START` and `LIMIT` to shard the manifest. Keep `SEED=10` for the main single-seed table; the multiseed scripts use 10, 11, and 12.

## 4. Baselines

External baseline code is intentionally not vendored. The adapters are:

- `scripts/run_samflow_baseline.py`
- `scripts/run_flowalign_baseline.py`
- `scripts/run_flux_persistent_baselines.py`

Point each adapter at a separately installed upstream repository/environment. `scripts/verify_baseline_envs.py` and `scripts/smoke_baseline_entrypoints.py` provide lightweight preflight checks. Preserve each upstream method's published settings and use the same input images, target prompts, seeds, and output resolution.

## 5. Metric assembly and evaluation

`prepare_flowedit135_metric_runs.py` assembles method outputs into the expected evaluation tree. Then run:

```bash
python scripts/evaluate_paper_metrics.py --help
python scripts/summarize_metrics_csv.py --help
```

The release retains the main comparison and one compact aggregate for each
ablation under:

- `results/submission/`
- `results/ablations/multiseed/`
- `results/ablations/components/`
- `results/ablations/masked_gap/`
- `results/ablations/pareto/`
- `results/ablations/t4_operator/`

The evaluation command regenerates per-family, per-seed, and audit files.
Reproduced aggregates should be compared against the corresponding checked-in
summary CSVs.

## 6. Regression checks

Run on a compute node with the target CUDA/PyTorch environment:

```bash
python -m pytest -q tests
python -m py_compile *.py flux/*.py scripts/*.py
```

The core refactor was additionally checked by bitwise output regression on A100 before this release cleanup. The cleanup itself changes repository organization and path handling, not controller math.
