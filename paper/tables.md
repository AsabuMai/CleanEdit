# Tables

Current source of truth:

```text
experiments/norestore_tradeoff/summary_main_no_samflow.csv
experiments/norestore_tradeoff/summary_by_method.csv
experiments/norestore_metrics/metrics.csv
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv
experiments/flowedit135_fixedmask_metrics_20260621/metric_audit.json
```

`summary_main_no_samflow.csv` is the main-table source. It is derived from
`summary_by_method.csv` by filtering out `sam_flow_sd3` and `sam_flow_flux`.
SAM-Flow is treated as concurrent arXiv context rather than a main
peer-reviewed baseline.

## Main Table

Recommended title:

```text
Main quantitative comparison on FlowEdit-135.
```

Scope:

```text
135 tasks x 12 main-table methods x seed 10 = 1620 completed metric rows.
```

Required table columns:

- `Method`
- `Backbone / family`
- `local_clip_t` up
- `clip_direction_similarity` up
- `bg_lpips` down
- `bg_l1` down
- `bg_dino_source` up
- `bg_ssim_luma` up

Optional columns:

- `clip_target_minus_source` up
- `clip_image_source_similarity` up
- `source_l1` down

Do not make `inside_l1` a main quality metric. It measures change magnitude in
the edit region, not edit correctness.

## Main Table Values

| Method | n | local CLIP-T up | CLIP dir up | BG-LPIPS down | BG-L1 down | BG-DINO up | BG-SSIM up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours-FLUX | 135 | 0.2480 | 0.2705 | **0.0145** | 0.0042 | **0.9988** | 0.9914 |
| Ours-SD3 | 135 | 0.2408 | 0.2697 | 0.0180 | **0.0028** | 0.9985 | **0.9939** |
| SplitFlow-SD3 | 135 | **0.2517** | **0.2864** | 0.0789 | 0.0380 | 0.9614 | 0.8869 |
| FlowEdit-SD3 | 135 | 0.2500 | 0.2798 | 0.0793 | 0.0383 | 0.9609 | 0.8862 |
| FireFlow | 135 | 0.2516 | 0.2402 | 0.1418 | 0.0578 | 0.9330 | 0.7851 |
| ReFLEx | 135 | 0.2511 | 0.2560 | 0.1098 | 0.0409 | 0.9613 | 0.7721 |
| RF-Solver-Edit | 135 | 0.2451 | 0.2357 | 0.1132 | 0.0400 | 0.9521 | 0.8236 |
| OT-RF enhanced-SD3 | 135 | 0.2295 | 0.2287 | 0.0296 | 0.0206 | 0.9846 | 0.9327 |
| InstructPix2Pix | 135 | 0.2367 | 0.1734 | 0.1063 | 0.0729 | 0.9621 | 0.7835 |
| LEDITS++ | 135 | 0.2433 | 0.2158 | 0.1241 | 0.0650 | 0.9108 | 0.7815 |
| FlowEdit-FLUX | 135 | 0.2430 | 0.2484 | 0.1236 | 0.0579 | 0.9438 | 0.8445 |
| DRFS-SD3 | 135 | 0.2208 | 0.1689 | 0.2833 | 0.1212 | 0.9128 | 0.5856 |

Suggested caption:

```text
We report edit strength with local CLIP-T and CLIP direction, and preservation
with metrics computed on a background region obtained by dilating the fixed
edit evaluation region and scoring its complement. The main Ours variants are
no-final-restore runs with no final pixel-level source compositing. Ours-FLUX
achieves the best BG-LPIPS and BG-DINO scores, Ours-SD3 achieves the best
BG-L1 and BG-SSIM scores, and SplitFlow-SD3 gives the strongest main-table
edit scores with larger non-edit drift.
```

## Trade-Off Figure Companion

Use the main table together with:

```text
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.pdf
```

Refresh or filter the figure before submission so `sam_flow_sd3` and
`sam_flow_flux` are not shown in the main paper plot. The figure should show
that the no-final-restore Ours variants lie in the high-preservation region of
the edit-preservation trade-off.

## Family Breakdown Table

Supplemental table source:

```text
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
```

This older family summary is for the restored/full baseline snapshot. Rebuild
or filter a no-final-restore family table before using family ranks in the
main manuscript.

Recommended appendix columns:

- `family_label`
- `method`
- `local_clip_t`
- `bg_lpips`
- `bg_l1`
- `bg_dino_source`
- `bg_ssim_luma`

Family-level message:

```text
Ours ranks first on preservation metrics in each of T1-T5, but its edit score
rank varies, especially on T3 surface-decal text-like edits.
```

Do not reuse this sentence verbatim until the no-final-restore family summary
has been regenerated.

## Metric Audit Table

Include this as a reproducibility sentence rather than a full table:

```text
All 1620 main-table rows are complete after excluding SAM-Flow. The
no-final-restore Ours rows are 270/270 complete, and the peer-reviewed baseline
rows are inherited from the 2026-06-21 FlowEdit-135 metric pass. All rows use
fixed evaluation regions and local target prompts. The fixed-region audit
contains 131 detected regions and 4 manual text-panel boxes.
```

## Concurrent ArXiv Context

SAM-Flow is a recent concurrent arXiv preprint with a source-anchored masked
flow design. Keep `sam_flow_sd3` and `sam_flow_flux` out of the main
peer-reviewed baseline table. If needed, report them in an appendix or related
work context table using the rows already present in
`experiments/norestore_tradeoff/summary_by_method.csv`.

## Archived Phase2 Tables

The older Phase2 tables remain in:

```text
experiments/support_v3_2026-06-02/
paper/phase2_experiment_report_2026-06-11.md
```

Do not mix Phase2 table numbers with the FlowEdit-135 main result unless the
manuscript explicitly frames Phase2 as an internal ablation or development
diagnostic.
