# Figures

Current FlowEdit-135 figure set:

```text
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.pdf
experiments/norestore_tradeoff/tradeoff_by_family_edit_preservation.png
experiments/norestore_tradeoff/tradeoff_by_family_edit_preservation.pdf
experiments/flowedit135_fixedmask_metrics_20260621/review_selected_methods.jpg
experiments/flowedit135_fixedmask_metrics_20260621/review_hard_ours_vs_sam.jpg
data/flowedit_compatible_135/eval_masks/_contact_sheet.jpg
```

The older `paper/assets/tradeoff_*` files were built from the restored/full
snapshot and should be refreshed before being used as main-paper assets.

## Main Figure: Edit-Preservation Trade-Off

Use:

```text
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png
```

Suggested caption:

```text
Edit-preservation trade-off on FlowEdit-135. The x-axis measures local edit
strength with local CLIP-T, and the y-axis measures non-edit-region
preservation as -log10(BG-LPIPS), where BG is the complement of a dilated fixed
evaluation region. The main Ours variants use no final pixel-level source
compositing and lie in the high-preservation region of the trade-off, while
SplitFlow-SD3 and other editing baselines produce stronger local edit scores at
the cost of higher background drift.
```

Markdown preview:

![Edit-preservation trade-off](../experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png)

Before submission, refresh or filter the main figure so SAM-Flow is not shown
in the main paper plot.

## Qualitative Comparison Figure

Use:

```text
experiments/flowedit135_fixedmask_metrics_20260621/review_selected_methods.jpg
```

This figure is useful for internal selection and appendix material. For the
main paper, crop it to fewer methods:

```text
source | fixed eval region | Ours-SD3 | Ours-FLUX | FlowEdit-SD3 | SplitFlow-SD3 | OT-RF enhanced
```

Recommended examples:

- `fe_046_cat_crown_1_black_top_hat`
- `fe_094_dog_6_red_top_hat`
- `fe_208_pizza_tomato_olive_2_mushrooms`
- `fe_024_bus_2_volkswagen_logo`
- `fe_118_gas_station_1_cvpr`
- `fe_000_bear_1_black_bear`
- `fe_221_rocks_6_colorful_wooden_blocks`
- `fe_171_meditation_1_wooden_statue`

Use the figure to show the qualitative trade-off: Ours preserves source
appearance and background tightly; stronger editing baselines often alter more
of the image.

## Hard-Case Figure

Use:

```text
experiments/flowedit135_fixedmask_metrics_20260621/review_hard_ours_vs_sam.jpg
```

This should not be a main positive-only figure. It is valuable for limitation
discussion because it shows that text-like T3 surface-decal edits are often
under-edited by Ours, while more aggressive editing baselines rewrite them more
clearly at the cost of larger non-edit drift. Because SAM-Flow is concurrent
arXiv context rather than a main baseline, do not use a SAM-Flow-centered hard
case figure as the main limitation figure.

## Family-Level Trade-Off Figure

Use:

```text
experiments/norestore_tradeoff/tradeoff_by_family_edit_preservation.png
```

Recommended appendix caption:

```text
Family-level edit-preservation trade-off. Orange stars denote Ours and grey
dots denote baselines. Ours consistently occupies the high-preservation region
for T1-T5, while edit strength varies by family.
```

As with the overall trade-off figure, refresh or filter this before submission
so SAM-Flow does not appear in the main-paper figure set.

## Fixed Evaluation Region Audit

Use:

```text
data/flowedit_compatible_135/eval_masks/_contact_sheet.jpg
data/flowedit_compatible_135/eval_masks/_audit.csv
```

This belongs in the appendix or reviewer-response material. It documents that
all 135 tasks have fixed evaluation regions. The regions are not claimed to be
exact segmentation masks.

## Archived Phase2 Figures

The older Phase2 figures remain useful for internal method history but should
not be used as the main benchmark result:

```text
paper/assets/phase2_main_qual_grid_seed12_2026-06-11.png
paper/assets/phase2_result_figure2_table1_metrics.png
paper/assets/phase2_external_baseline_bars_sd3.png
paper/assets/phase2_external_baseline_bars_flux_context.png
paper/assets/phase2_result_figure4_family_preservation.png
paper/assets/phase2_result_figure5_proxy_audit.png
```
