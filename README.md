# CleanEdit

CleanEdit is a training-free image editor for rectified-flow diffusion models. It combines operation-conditioned spatial support with clean-space adaptive control, using a shared controller design for SD3 and FLUX while keeping backend-specific model plumbing separate.

This repository is the compact reproduction release. Generated images, model weights, external baseline clones, caches, and exploratory runs are intentionally excluded. The complete pre-cleanup laboratory snapshot is preserved in the Git tag `archive/full-lab-2026-07-14`.

## What is included

```text
CleanEdit/
├── run_edit_sd3.py              # single-image SD3 entry point
├── run_edit_flux.py             # single-image FLUX entry point
├── dece_core.py                 # shared controller and clean-control logic
├── operation_support_v3.py      # operation-conditioned support
├── sd3_*.py                     # SD3 backend
├── flux/                        # FLUX backend
├── scripts/                     # batch, metric, and baseline adapters
├── slurm/                       # cluster launchers
├── data/flowedit_compatible_135 # benchmark manifests and fixed masks
├── results/                     # published summaries and audits
├── tests/                       # controller and metric regression tests
└── docs/                        # method and reproduction notes
```

## Installation

Python 3.10+ and a CUDA-capable PyTorch installation are expected. Create the environment in the repository so batch jobs and interactive runs use the same packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

Model weights are downloaded from Hugging Face on first use. Accept the corresponding model licenses and authenticate beforehand when required. SD3 uses `stabilityai/stable-diffusion-3-medium-diffusers`; the FLUX backend uses the model configured by `flux/flux_hrec.py`.

## Single-image editing

The complete argument surface is available with `--help`. Minimal commands are:

```bash
source .venv/bin/activate
python run_edit_sd3.py \
  --image input.png \
  --source-prompt "a red mug on a table" \
  --prompt "a blue mug on a table" \
  --output outputs/sd3.png

python run_edit_flux.py \
  --image input.png \
  --source-prompt "a red mug on a table" \
  --prompt "a blue mug on a table" \
  --output outputs/flux.png
```

The benchmark batch launchers contain the operation-specific settings used for the reported runs.

## Reproduce FlowEdit-135

1. Obtain the source images from the official FlowEdit repository according to its license.
2. Rebase the checked-in manifest to that local image directory:

```bash
python scripts/rebase_manifest.py \
  --manifest data/flowedit_compatible_135/manifest.json \
  --flowedit-root /path/to/FlowEdit \
  --output data/flowedit_compatible_135/manifest.local.json
```

3. Launch SD3 or FLUX. On Slurm:

```bash
MANIFEST=$PWD/data/flowedit_compatible_135/manifest.local.json \
  sbatch slurm/run_sd3_flowedit135.sbatch

MANIFEST=$PWD/data/flowedit_compatible_135/manifest.local.json \
  sbatch slurm/run_flux_flowedit135.sbatch
```

4. Assemble runs and evaluate them with the scripts documented in [docs/reproduction.md](docs/reproduction.md).

## Fair comparison protocol

The main comparison uses the same benchmark inputs, masks, seeds, resolution, and metric implementation for all methods. CleanEdit's main SD3/FLUX launchers set `ABLATE=no_final_postprocess`: the reported image is the diffusion result, with no final source-pixel restoration, cut-and-paste, or method-specific compositing. Diagnostic post-processing modes remain in the implementation for ablation and debugging only and must not be mixed into the main comparison.

## Results and limitations

Compact CSV/JSON summaries and metric audits are in [`results/`](results/). The known limitations include unreliable small rendered text in SD3 and incomplete whole-subject material conversion for large subjects; see [paper/limitations.md](paper/limitations.md).

## Tests

Run tests on a CUDA compute node when the environment imports GPU-enabled PyTorch:

```bash
python -m pytest -q tests
```

## Citation

The paper has been submitted. Citation metadata will be added when a public paper record is available.
