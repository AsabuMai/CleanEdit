# Tables

Current source of truth:

```text
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv
experiments/flowedit135_fixedmask_metrics_20260621/metric_audit.json
```

## Main Table

Recommended title:

```text
Main quantitative comparison on FlowEdit-135.
```

Scope:

```text
135 tasks x 13 methods x seed 10 = 1755 completed metric rows.
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
| Ours-SD3 | 135 | 0.2447 | 0.2757 | **0.0054** | **0.0029** | **0.9984** | **0.9939** |
| SAM-Flow-SD3 | 135 | **0.2543** | **0.3083** | 0.0363 | 0.0236 | 0.9745 | 0.9106 |
| SplitFlow-SD3 | 135 | 0.2517 | 0.2864 | 0.0789 | 0.0380 | 0.9614 | 0.8869 |
| FlowEdit-SD3 | 135 | 0.2500 | 0.2798 | 0.0793 | 0.0383 | 0.9609 | 0.8862 |
| SAM-Flow-FLUX | 135 | 0.2238 | 0.2302 | 0.0179 | 0.0164 | 0.9875 | 0.9479 |
| OT-RF enhanced-SD3 | 135 | 0.2295 | 0.2287 | 0.0296 | 0.0206 | 0.9846 | 0.9327 |
| FireFlow | 135 | 0.2516 | 0.2402 | 0.1418 | 0.0578 | 0.9330 | 0.7851 |
| RF-Solver-Edit | 135 | 0.2451 | 0.2357 | 0.1132 | 0.0400 | 0.9521 | 0.8236 |
| ReFLEx | 135 | 0.2511 | 0.2560 | 0.1098 | 0.0409 | 0.9613 | 0.7721 |
| InstructPix2Pix | 135 | 0.2367 | 0.1734 | 0.1063 | 0.0729 | 0.9621 | 0.7835 |
| LEDITS++ | 135 | 0.2433 | 0.2158 | 0.1241 | 0.0650 | 0.9108 | 0.7815 |
| FlowEdit-FLUX | 135 | 0.2430 | 0.2484 | 0.1236 | 0.0579 | 0.9438 | 0.8445 |
| DRFS-SD3 | 135 | 0.2208 | 0.1689 | 0.2833 | 0.1212 | 0.9128 | 0.5856 |

Suggested caption:

```text
We report edit strength with local CLIP-T and CLIP direction, and preservation
with metrics computed on a background region obtained by dilating the fixed
edit evaluation region and scoring its complement. Ours achieves the strongest
background preservation while maintaining non-trivial edit signal, whereas
SAM-Flow-SD3 gives the strongest edit scores with larger non-edit drift.
```

## Trade-Off Figure Companion

Use the main table together with:

```text
paper/assets/tradeoff_overall_edit_preservation_compact.png
paper/assets/tradeoff_overall_edit_preservation_compact.pdf
```

The figure shows that Ours-SD3 lies on the edit-preservation Pareto frontier.

## Family Breakdown Table

Supplemental table source:

```text
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
```

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

## Metric Audit Table

Include this as a reproducibility sentence rather than a full table:

```text
All 1755 rows are complete. All rows use fixed evaluation regions and local
target prompts. The fixed-region audit contains 131 detected regions and 4
manual text-panel boxes.
```

## Archived Phase2 Tables

The older Phase2 tables remain in:

```text
experiments/support_v3_2026-06-02/
paper/phase2_experiment_report_2026-06-11.md
```

Do not mix Phase2 table numbers with the FlowEdit-135 main result unless the
manuscript explicitly frames Phase2 as an internal ablation or development
diagnostic.
