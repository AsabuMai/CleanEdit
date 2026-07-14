# Paper Materials

The submitted paper scope is the FlowEdit-135 benchmark-style evaluation, not the
older Phase2 15-case diagnostic set.

This directory is frozen paper-facing material. See `../docs/SUBMISSION_FREEZE.md`;
do not silently replace its evidence with nine-bucket or later post-submission
outputs.

Use these current files:

- `results.md`
- `tables.md`
- `figures.md`
- `limitations.md`
- `wacv_experiment_design.md`
- `flowedit135_experiment_report_2026-06-21.md`
- `references.bib`

Older Phase2 materials remain useful for method development history, ablations,
and appendix context, but they should not be treated as the main paper evidence.

## Active Evidence

- `data/flowedit_compatible_135/manifest.json`
- `data/flowedit_compatible_135/local_target_prompts.json`
- `data/flowedit_compatible_135/eval_masks/`
- `outputs/flowedit135_metric_runs/`
- `outputs/norestore_metric_runs/`
- `experiments/norestore_metrics/metrics.csv`
- `experiments/norestore_tradeoff/summary_main_no_samflow.csv`
- `experiments/norestore_tradeoff/summary_by_method.csv`
- `experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv`
- `experiments/flowedit135_fixedmask_metrics_20260621/summary_by_method.csv`
- `experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv`
- `experiments/flowedit135_fixedmask_metrics_20260621/metric_audit.json`
- `experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png`
- `experiments/norestore_tradeoff/tradeoff_by_family_edit_preservation.png`

## Main Claim

The evidence supports a preservation-first claim:

```text
Our method achieves the strongest non-edit-region preservation on FlowEdit-135
while remaining on the edit-preservation Pareto frontier.
```

The paper should not claim that the method is the most aggressive editor or that
it wins every edit-success metric. The main table uses the no-final-restore
variant of our method and excludes SAM-Flow from the peer-reviewed baseline set
because SAM-Flow is a recent concurrent arXiv preprint with a closely related
source-anchored masked-flow design. Keep SAM-Flow only as optional concurrent
context, not as a main-table baseline.

## Evaluation Snapshot

- Dataset: FlowEdit-compatible 135 tasks.
- Task families: T1 attached accessory, T2 container insertion, T3 surface
  decal, T4 local recolor, T5 same-color material.
- Main-table methods: 12 total, including Ours-SD3, Ours-FLUX, FlowEdit,
  SplitFlow, FireFlow, RF-Solver-Edit, ReFLEx, InstructPix2Pix, LEDITS++,
  OT-RF enhanced, and DRFS.
- Main-table rows: 1620/1620 = 135 tasks x 12 methods x seed 10.
- No-final-restore Ours rows: 270/270 = Ours-SD3 and Ours-FLUX on 135 tasks.
- Fixed evaluation regions: 135/135.
- Local target prompts: 135/135.

## Writing Notes

- Use `local_clip_t` and CLIP direction as edit-strength evidence.
- Use `bg_lpips`, `bg_l1`, `bg_dino_source`, and `bg_ssim_luma` as the main
  preservation evidence.
- Refer to masks as fixed evaluation regions, not exact segmentation masks.
- State that the main Ours variants do not use final pixel-level source
  compositing; they are no-final-restore runs.
- Treat SAM-Flow as optional concurrent arXiv context, not as a main benchmark
  baseline.
- Put text-editing and very large fixed-region cases in limitations or appendix.
