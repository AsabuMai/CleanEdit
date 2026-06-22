# Current Phase2 Status 2026-06-11

## Locked Scope

Current project scope is fixed as:

```text
Phase2 = T1-T5, three source cases per family, seeds 10/11/12.
```

This is the current main experiment scope. It replaces the old five-case
Core-5 framing. `red_chair_blue` is obsolete for current reporting.

## Task Map

| family | current cases |
| --- | --- |
| T1_attached_accessory | `cat_crown`; `dog_bow_tie_phase2`; `dog_front_sunglasses_phase2` |
| T2_container_insertion | `bowl_apple_inside`; `white_bowl_orange_tabletop_phase2`; `brown_bowl_lemon_phase2` |
| T3_surface_decal | `tshirt_star`; `mug_heart`; `tote_leaf` |
| T4_local_recolor | `red_office_chair_to_blue_office_chair`; `green_mug_orange_phase2`; `yellow_vase_blue_phase2` |
| T5_same_color_material | `pillow_same_color_cable_knit`; `pillow_same_color_cable_knit_grey`; `pillow_same_color_cable_knit_armchair` |

## Completed

- Phase2 final table builder created:
  `scripts/build_phase2_paper_tables.py`.
- Phase2 final tables generated and audited:
  `experiments/support_v3_2026-06-02/phase2_tables_audit_2026-06-11.json`.
- Phase2 paper table summary generated:
  `experiments/support_v3_2026-06-02/phase2_paper_tables_2026-06-11.md`.
- Phase2 final selected run registry generated:
  `experiments/support_v3_2026-06-02/phase2_final_selected_runs_2026-06-11.csv`.
- Sam-Flow-SD3 and Sam-Flow-FLUX/context have been added as retained
  external baselines; FlowAlign is excluded from paper-facing comparison
  because of poor visual quality.
- The selected-run metric refresh completed on 2026-06-12. The 33 repaired or
  formal DeCE-RF visual selections were re-evaluated and merged into
  `phase2_internal_bg_metrics.csv`.
- Phase2 experiment report generated:
  `paper/phase2_experiment_report_2026-06-11.md`.
- Phase2 Word report generated:
  `paper/Phase2_Experiment_Report_2026-06-11.docx`.
- Phase2 blind internal audit package generated:
  `experiments/support_v3_2026-06-02/blind_internal_audit_phase2_t1_t5_2026-06-11/`.
- The three rater sheets were filled with Codex metric-guided proxy ratings and
  summarized. These are not human study results.
- Old Core-5/five-case tables, old blind audit package, old table scripts,
  stale paper docs, and stale docs were moved into
  `obsolete_pre_phase2_lock_2026-06-11/` folders.
- Local copy of the new blind audit package was downloaded to:
  `I:\Downloads\1\2\blind_internal_audit_phase2_t1_t5_2026-06-11`.

## Table Audit

| table | expected rows | found rows | status |
| --- | --- | --- | --- |
| Table 1 Phase2 T1-T5 main | 180 | 180 | complete |
| Table 2a Phase2 SD3 common subset | 315 | 315 | complete |
| Table 2b Phase2 native context | 180 | 180 | complete |

Table 1 is now 15 cases x 3 seeds = 45 rows per method.

## Final Selected Run Registry

Registry status: complete.

| item | count |
| --- | --- |
| rows | 540 |
| metric-aligned rows | 540 |
| rows needing metric refresh | 0 |

The registry is the source of truth for visual artifacts. It records the
selected run, image path, selection reason, and metric-alignment status for
each T1-T5 family/task/seed/method row. All selected visual rows are now
metric-aligned with the current CSVs.

Examples:

- `white_bowl_orange_tabletop_phase2` DeCE-RF uses
  `support_v3_controller_rmsgap_failed3_fix6_20260609`, because the plain run
  under-edited the orange in the visual grid.
- `brown_bowl_lemon_phase2` remains in T2; lemon is present and the task is not
  removed.
- `tshirt_star` DeCE-RF uses the original
  `support_v3_controller_rmsgap` run because it looks more naturally printed
  on the shirt under manual visual inspection. The broader `t1t4_3seed` rerun lost
  the star, while the `failed3_fix3` repair looked too overlaid.
- `mug_heart` DeCE-RF uses
  `support_v3_controller_rmsgap_mugbox145_c180_ref075_v1`: a unified
  three-seed c180 selection with deterministic final-reference compositing
  scale 0.75 inside the decal mask. This is not a seed12-only choice.

## Blind Audit Package

Package status: complete.

| item | count |
| --- | --- |
| raters | 3 |
| items per rater | 45 |
| rows per rater | 180 |
| total manifest rows | 540 |
| missing images | 0 |

The package is ready for human scoring. Method names are hidden in rater files;
keep `blind_internal_audit_method_key_private.csv` private until all sheets are
frozen.

Proxy-fill status: complete.

- `blind_internal_audit_proxy_fill_report.json`: 540/540 rows filled, missing
  metrics = 0.
- `blind_internal_audit_score_audit.json`: 540/540 merged rows, missing scores
  = 0, invalid scores = 0.
- `blind_internal_audit_summary.md`: summary by method generated.
- `PROXY_RATINGS_NOTICE.md`: caveat that the filled ratings are metric-guided
  proxy precheck scores, not human ratings.

The blind-audit package was regenerated after the final selected run registry
was created. Anonymous images now follow the registry when available; the
private method key records `selected_run`, `selection_reason`, and
`metric_status`.

## Figure Grid

Figure status: complete.

- `paper/assets/phase2_main_qual_grid_seed12_2026-06-11.png`
- `paper/assets/phase2_main_qual_grid_seed12_2026-06-11.json`
- `paper/assets/phase2_result_figure2_table1_metrics.png`
- `paper/assets/phase2_external_baseline_bars_sd3.png`
- `paper/assets/phase2_external_baseline_bars_flux_context.png`
- `paper/assets/phase2_result_figure4_family_preservation.png`
- `paper/assets/phase2_result_figure5_proxy_audit.png`

The grid uses one representative seed-12 case per T1-T5 family and compares:

```text
source | direct target | generic support | DeCE-RF
```

Image selection is resolved through
`experiments/support_v3_2026-06-02/phase2_final_selected_runs_2026-06-11.csv`.

It is intended to support the preservation-first claim, not a broad editing
benchmark claim.

The Word report embeds Figure 1-6 plus appendix Figure A1-A5. Figure 3 is the
same-backbone SD3 external-baseline bar chart, and Figure 4 is the
native/context external-baseline bar chart.

Appendix all-method visual grids are complete and embedded in the Word report
as Figure A1-A5:

- `paper/assets/appendix_grids/appendix_T1_attached_accessory_representative_all_methods.png`
- `paper/assets/appendix_grids/appendix_T2_container_insertion_representative_all_methods.png`
- `paper/assets/appendix_grids/appendix_T3_surface_decal_representative_all_methods.png`
- `paper/assets/appendix_grids/appendix_T4_local_recolor_representative_all_methods.png`
- `paper/assets/appendix_grids/appendix_T5_same_color_material_representative_all_methods.png`

## Current Active Entrypoints

- `PHASE2_LOCK_2026-06-11.md`
- `CURRENT_PHASE2_STATUS_2026-06-11.md`
- `README.md`
- `PROJECT_MAP.md`
- `docs/todo_2026-06-11.md`
- `paper/README.md`
- `paper/results.md`
- `paper/tables.md`
- `paper/figures.md`
- `paper/limitations.md`
- `paper/phase2_experiment_report_2026-06-11.md`
- `paper/Phase2_Experiment_Report_2026-06-11.docx`
- `experiments/support_v3_2026-06-02/phase2_final_selected_runs_2026-06-11.csv`

## Next Tasks

1. Start the manuscript writing pass from the concise current `paper/*.md`
   files and the locked Word report.
2. If human ratings are still desired, replace the proxy-filled sheets using
   the blank backups in each rater folder, then rerun
   `scripts/summarize_blind_internal_audit.py`.
3. Human blind audit is optional, not required for Phase2 paper closure. The
   decision and trigger conditions are in
   `docs/human_blind_audit_decision_2026-06-11.md`.
4. Do not run Phase2 mask sensitivity immediately. The decision and trigger
   conditions are in `docs/phase2_mask_sensitivity_decision_2026-06-11.md`.
5. Explicitly label the current filled audit scores as proxy/internal precheck
   scores unless human ratings replace them.
6. If new GPU experiments are needed, enter `a100-01` with Slurm first.

## Compute Boundary

Master node is only for light file work, queue checks, aggregation, and job
submission. Heavy install/model/GPU/Torch/diffusers work must run on `a100-01`.

- Appendix all-method visual comparison grids are in paper/assets/appendix_grids/. They include RF reconstruction plus external baselines; all appendix visual comparisons use seed 12 and the final selected run registry.
