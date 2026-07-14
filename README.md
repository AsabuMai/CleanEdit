# CleanEdit

The paper has been submitted. Its evidence is frozen; start with
`docs/SUBMISSION_FREEZE.md`. Work performed after submission is indexed in
`post_submission/README.md` and must not be mixed into the submitted results.

Submitted paper method:

```text
CleanEdit: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing
```

Current paper scope:

```text
FlowEdit-compatible 135-task benchmark, seed 10, preservation-first evaluation.
```

The older Phase2 T1-T5 diagnostic set remains useful for method-development
history, ablations, and appendix context, but it is no longer the main paper
evidence. Start with the paper-facing files before using any Phase2 document.

## Active Entry Points

- `docs/SUBMISSION_FREEZE.md`: frozen submission evidence and boundary.
- `post_submission/README.md`: nine-bucket and later research status.
- `paper/README.md`: current paper scope, evidence, and claim boundary.
- `paper/results.md`: current quantitative result narrative.
- `paper/tables.md`: main table source and table policy.
- `paper/figures.md`: current figure set and figure cautions.
- `paper/limitations.md`: current limitation language.
- `PROJECT_MAP.md`: active file map.
- `docs/README.md`: docs index and current execution queue.
- `docs/todo_2026-06-11.md`: current queue, superseding the old Phase2 queue.

## Active Evidence

Current FlowEdit-135 artifacts:

```text
data/flowedit_compatible_135/manifest.json
data/flowedit_compatible_135/local_target_prompts.json
data/flowedit_compatible_135/eval_masks/
outputs/flowedit135_metric_runs/
outputs/norestore_metric_runs/
experiments/norestore_metrics/metrics.csv
experiments/norestore_tradeoff/summary_by_method.csv
experiments/norestore_tradeoff/summary_main_no_samflow.csv
experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/metric_audit.json
```

Current figure artifacts:

```text
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png
experiments/norestore_tradeoff/tradeoff_by_family_edit_preservation.png
experiments/flowedit135_fixedmask_metrics_20260621/review_selected_methods.jpg
data/flowedit_compatible_135/eval_masks/_contact_sheet.jpg
```

## Active Scripts

- `prepare_flowedit135_metric_runs.py`
- `scripts/evaluate_paper_metrics.py`
- `scripts/pie_sd3_batch_kindaware.py`
- `scripts/pie_batch_flux_kindaware_v11.py`
- `scripts/run_samflow_baseline.py`
- `scripts/allpass_v1/build_allpass_v1.py`
- `scripts/allpass_v1/run_allpass_batch.py`

## Historical Phase2 Files

These are historical/internal diagnostic entry points, not the main paper
scope:

- `PHASE2_LOCK_2026-06-11.md`
- `CURRENT_PHASE2_STATUS_2026-06-11.md`
- `experiments/support_v3_2026-06-02/`
- `paper/phase2_experiment_report_2026-06-11.md`

Do not mix Phase2 table numbers with the FlowEdit-135 main result unless the
manuscript explicitly frames Phase2 as an internal ablation or development
diagnostic.

## Compute Boundary

Do not run heavy operations on the master node. Heavy install/model/GPU/Torch/
diffusers work must run through Slurm. Use `a100-01` for SD3 work:

```bash
srun -p a100 -w a100-01 --gres shard:1 --pty /bin/bash -l
```

Use `h100-01` for FLUX work; FLUX jobs must request the `h100` partition and
verify the hostname before loading the model.

The shared environment is:

```text
/cluster/users/grad/2025/25t8103/project/.venv
```

## Current Claim

The supported claim is preservation-first:

```text
On FlowEdit-135, CleanEdit achieves the strongest non-edit-region preservation
among the main peer-reviewed baseline set while remaining on the
edit-preservation Pareto frontier.
```

Do not claim strongest edit amplitude, broad image-editing SOTA, or robust text
replacement. The main Ours rows are no-final-restore runs and do not use final
pixel-level source compositing.
