# RF h-Edit Project Map

The submitted FlowEdit-135 paper evidence is frozen as documented in
`docs/SUBMISSION_FREEZE.md`. Post-submission work is indexed separately in
`post_submission/README.md`.

## Current Scope

```text
FlowEdit-compatible 135 tasks, T1-T5 families, seed 10.
```

The current main table uses 12 peer-reviewed/main-table methods after excluding
SAM-Flow from the main baseline set as concurrent arXiv context. The complete
main-table scope is:

```text
135 tasks x 12 methods x seed 10 = 1620 metric rows.
```

The main Ours rows are the no-final-restore Ours-SD3 and Ours-FLUX runs:

```text
270/270 Ours rows complete.
```

## Active Paper Sources

- `paper/README.md`: current paper scope and claim boundary.
- `paper/results.md`: result narrative and interpretation.
- `paper/tables.md`: main table values, captions, and table cautions.
- `paper/figures.md`: figure sources and figure refresh cautions.
- `paper/limitations.md`: current limitation language.
- `paper/wacv_experiment_design.md`: benchmark/design context.
- `paper/flowedit135_experiment_report_2026-06-21.md`: earlier FlowEdit-135
  report retained for traceability.
- `paper/references.bib`: bibliography.

## Active Code

- `run_edit_sd3.py`: main SD3 RF editing CLI.
- `run_edit_flux.py`: FLUX entry shim.
- `sd3_hrec.py`: RF editing orchestration and ODE controller.
- `sd3_model_ops.py`: SD3 velocity calls, Q/K/V injection, feature extraction,
  x0 prediction, and source inversion.
- `flux/flux_hrec.py`: FLUX orchestration and ODE controller.
- `flux/flux_model_ops.py`: FLUX model operations.
- `flux/dece_flux_adapter.py`: packed-token adapter into `dece_core.py`.
- `dece_core.py`: backend-neutral edit/preserve velocity core used by FLUX.
  SD3 currently mirrors the same energy composition inline in `sd3_hrec.py`
  An opt-in `--shared-core-shadow` path checks numerical equivalence without
  changing SD3 outputs; see `docs/shared_core_shadow_20260713.md`.
- `operation_support_v3.py`: operation-aware support proposal.
- `edit_preprocess.py`: image preprocessing, normalized box helpers, and mask
  builders used by the CLI.
- `spatial_masks.py`: mask morphology, support fusion, mask statistics, and
  external mask/image loading.
- `guidance_fields.py`: decode-space color/reference losses and guidance-field
  smoothing/projection/RMS limiting.
- `recolor_projection.py`: recolor clean-projection and debug helpers.

## Active Evaluation And Runners

- `prepare_flowedit135_metric_runs.py`
- `scripts/evaluate_paper_metrics.py`
- `scripts/pie_sd3_batch_kindaware.py`
- `scripts/pie_batch_flux_kindaware_v11.py`
- `scripts/run_samflow_baseline.py`
- `scripts/allpass_v1/build_allpass_v1.py`
- `scripts/allpass_v1/run_allpass_batch.py`
- `scripts/allpass_v1/slurm/`

## Active Data And Experiments

Dataset and fixed evaluation regions:

```text
data/flowedit_compatible_135/manifest.json
data/flowedit_compatible_135/local_target_prompts.json
data/flowedit_compatible_135/eval_masks/
```

Main no-final-restore Ours metrics:

```text
outputs/norestore_metric_runs/
experiments/norestore_metrics/metrics.csv
experiments/norestore_metrics/metrics.json
```

Current main-table summaries:

```text
experiments/norestore_tradeoff/summary_by_method.csv
experiments/norestore_tradeoff/summary_main_no_samflow.csv
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png
experiments/norestore_tradeoff/tradeoff_by_family_edit_preservation.png
```

Baseline metric pass:

```text
outputs/flowedit135_metric_runs/
experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/metric_audit.json
```

## Historical / Internal Diagnostics

- `PHASE2_LOCK_2026-06-11.md`: locked Phase2 diagnostic scope.
- `CURRENT_PHASE2_STATUS_2026-06-11.md`: completed Phase2 diagnostic status.
- `experiments/support_v3_2026-06-02/`: Phase2 internal evidence and tables.
- `paper/phase2_experiment_report_2026-06-11.md`: historical Phase2 report.
- `obsolete_pre_phase2_lock_2026-06-11/`: pre-lock archives.

Phase2 artifacts may be cited as method-development history or appendix
diagnostics, but not as the current main benchmark result.

## Post-submission research

- `post_submission/nine_bucket/`: July 4–6 visual-defect campaign.
- `post_submission/diagnostics/2026-07-13/`: shared-controller and fairness
  probes.
- `docs/fix_log_20260704.md`: detailed mechanism and limitation record.

Post-submission results are not part of the submitted paper evidence.

## Cleanup Rule

Use FlowEdit-135 files for current reporting. Do not treat the old Phase2
T1-T5 lock, the Core-5 rows, or `red_chair_blue` as current main-paper
evidence. Keep SAM-Flow out of the main peer-reviewed baseline table unless it
is explicitly labeled as concurrent context.

Legacy pre-submission Slurm launchers remain at the repository root because old
job notes refer to those paths. Do not add new launchers there. New work belongs
under `post_submission/<campaign>/slurm/`; move a legacy launcher only when its
references have been checked.
