# Paper Materials

Current paper scope is now the FlowEdit-135 benchmark-style evaluation, not the
older Phase2 15-case diagnostic set.

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
- `experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv`
- `experiments/flowedit135_fixedmask_metrics_20260621/summary_by_method.csv`
- `experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv`
- `experiments/flowedit135_fixedmask_metrics_20260621/metric_audit.json`
- `paper/assets/tradeoff_overall_edit_preservation_compact.png`
- `paper/assets/tradeoff_by_family_edit_preservation.png`

## Main Claim

The evidence supports a preservation-first claim:

```text
Our method achieves the strongest non-edit-region preservation on FlowEdit-135
while remaining on the edit-preservation Pareto frontier.
```

The paper should not claim that the method is the most aggressive editor or that
it wins every edit-success metric. The data show a clear trade-off: SAM-Flow-SD3
and related baselines can produce stronger local CLIP edit scores, while our
method changes the background far less.

## Evaluation Snapshot

- Dataset: FlowEdit-compatible 135 tasks.
- Task families: T1 attached accessory, T2 container insertion, T3 surface
  decal, T4 local recolor, T5 same-color material.
- Methods: 13 total, including Ours, FlowEdit, SplitFlow, SAM-Flow, FireFlow,
  RF-Solver-Edit, ReFLEx, InstructPix2Pix, LEDITS++, OT-RF enhanced, and DRFS.
- Completed runs: 1755/1755.
- Fixed evaluation regions: 135/135.
- Local target prompts: 135/135.

## Writing Notes

- Use `local_clip_t` and CLIP direction as edit-strength evidence.
- Use `bg_lpips`, `bg_l1`, `bg_dino_source`, and `bg_ssim_luma` as the main
  preservation evidence.
- Refer to masks as fixed evaluation regions, not exact segmentation masks.
- Put text-editing and very large fixed-region cases in limitations or appendix.
