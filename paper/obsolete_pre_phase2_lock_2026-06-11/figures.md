# Figure Plan

Current source of truth: `paper/wacv_experiment_design.md`.

The main paper should use five figures by default and six at most. Target
about 50-70 result image cells total. More than that starts to read as a
gallery and weakens the algorithmic story.

Completed legacy server evidence is documented in `paper/archive_old_core6_20260602/old_core6_server_results.md`.
The active strict headline evidence is the Core-5 Phase 1 matrix under
`experiments/support_v3_2026-06-02/`: attached accessory, container-constrained
spatial insertion, surface decal, local recolor, and same-color material
replacement. Simple exposed-object removal is an E5 boundary probe, not a
headline E1 row. Use the archived server grids only as supplementary
diagnostics; they are not the source of truth for the updated strict T2/T5
rows.

Use only complete runs with `result.png`, `stats.json`, `metadata.json`, and
`command.txt`.

## Main-Paper Figure Budget

| Figure | Content | Approx. result cells | Role |
| --- | --- | ---: | --- |
| Figure 1 | teaser: two examples, Source/Target/Direct/Generic/CleanEdit | 10 | motivation |
| Figure 2 | method overview | 0 | explain the algorithm |
| Figure 3 | E1 Core-5 qualitative grid | 24-30 | main effect |
| Figure 4 | E3 support geometry ablation | 12-18 | support component evidence |
| Figure 5 | E4 controller / stress Pareto curve | 0 | controller evidence |
| Figure 6 | E5 boundary and failure probes | 12-18 | scope boundary |

Main-paper target:

```text
tight: 45-55 result image cells
complete: 60-75 result image cells
```

Supplement target:

```text
150-300 image cells: all seeds, full grids, support masks, RF baselines,
Pareto sweeps, and failure taxonomy.
```

## E1 Main Qualitative Grid

Primary qualitative grid:

```text
Source | Target/Instruction | Direct target | Generic support | CleanEdit
```

Updated strict Core-5 target rows:

```text
T1 attached accessory: cat_crown
T2 container-constrained spatial insertion: bowl_apple_inside
T3 surface decal/logo: tshirt_star
T4 local recolor: red_chair_blue
T5 localized same-color material replacement: pillow_same_color_cable_knit
E5 boundary probe: backpack_remove_toy_charm
```

Current generated strict evidence and audit grids:

```text
experiments/support_v3_2026-06-02/visual_audit/
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.csv
experiments/support_v3_2026-06-02/e1_core5_strict_metrics.csv
experiments/support_v3_2026-06-02/table1_core5_final.csv
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/mask_sensitivity_core5_summary.md
```

Paper-use guidance:

- `cat_crown`: use as the attached-accessory success case.
- `dog_sunglasses`: diagnostic only; CleanEdit eyewear placement is too high for a strong figure.
- `tshirt_star`: use as the strict surface-decal success case.
- `mug_heart`: diagnostic only; visually clean but too small/weak for the main grid.
- `bowl_apple_inside`: use as the strict T2 insertion row.
- `pillow_same_color_cable_knit`: use as the revised T5 same-color material replacement. Keep `pillow_same_color_corduroy_panel` and `pillow_vertical_fabric_strip` as diagnostic examples, not the main T5 figure.
- `backpack_remove_toy_charm`: use as exposed-object removal success with a
  caveat that global CLIP underestimates removal quality.
- `red_chair_blue`: server evidence supports it as the localized
  attribute/recolor row; do not claim general recoloring.
- `red_office_chair_to_blue_office_chair`: fallback only if `red_chair_blue`
  fails final visual audit.

Do not include `support_v3_fixed` in this main qualitative grid unless the
layout still fits. Fixed CleanEdit belongs mainly in the controller ablation figure
and table.

## Optional E2 Baseline Panel

Status: the strict same-backbone SD3 target-mode RF comparison is complete for
FlowEdit, FlowAlign, and SplitFlow. The redesigned E2 now treats this as E2.2,
not as the whole baseline story. E2.1 calibration, E2.3 native preservation-aware
RF rows, and E2.4 support-matched diagnostics are added to address backbone and
input-condition fairness.

E2 should be primarily table-driven: Table 2a for same-backbone SD3, Table 2b
for runnable native/context rows, and Supplement Table S1 for support-matched
diagnostics. Do not reserve Figure 4 for E2. If a qualitative E2 panel is
needed, put it in supplement or use an unnumbered compact inset.

Current E2 audit artifacts:

```text
experiments/support_v3_2026-06-02/e2_baseline_download_registry.csv
experiments/support_v3_2026-06-02/e2_baseline_runnable_validation.csv
experiments/support_v3_2026-06-02/e2_baseline_audit.md
```

Current same-backbone SD3 artifacts:

```text
experiments/support_v3_2026-06-02/table2a_sd3_common_subset_final.csv
experiments/support_v3_2026-06-02/table2b_native_context_final.csv
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed10_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed11_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed12_grid.png
```

Preferred columns for a same-backbone SD3 supplement panel:

```text
Source | FlowEdit-SD3 | FlowAlign-SD3 or SplitFlow-SD3 | Fixed CleanEdit-SD3 | CleanEdit-SD3
```

Preferred columns for a native-context supplement panel:

```text
Source | FlowEdit-SD3 | Fixed CleanEdit-SD3 | Native RF row with backbone label | CleanEdit-SD3
```

Use two or three representative strict examples. Do not crowd the optional E2
panel with all baselines. The main quantitative E2 evidence is Table 2a.

Caption requirement:

```text
Backbone is shown in each method label. Same-backbone SD3 rows support the
algorithmic comparison; native-backbone rows, when included, are contextual.
```

Do not present this figure as a broad RF/FLUX victory claim.

## Figure 6 Boundary / Extension Probe Figure

Rows:

```text
laptop_remove_sticker
whiteboard_probe_red_star_sticker
```

Columns:

```text
Source | Base CleanEdit | Extension route | Support / gate annotation
```

Paper-use guidance:

- `laptop_remove_sticker`: show high-confidence completion clean-delta as a
  planar removal extension. Label it as `CleanEdit + completion prior`, not as
  the base CleanEdit method.
- `whiteboard_probe_red_star_sticker`: show non-glyph replacement in a
  semantic letter field. Label it as `CleanEdit + replacement route`.

## Figure 4 Support Geometry Figure

Recommended row:

```text
tshirt_star
```

Panels:

```text
attention evidence | clean disagreement | velocity disagreement |
operation-conditioned support | M_edit/M_core | M_preserve
```

## Figure 5 Feedback / Stress Pareto Figure

Use `cat_crown`, `tshirt_star`, or `pillow_same_color_cable_knit`, comparing:

```text
support_v3_fixed | CleanEdit across stress levels
```

Curves:

```text
edit-preserve Pareto frontier
edit target gap
preserve drift
adaptive edit weight
adaptive preserve weight
projection norm / ratio
```

Current artifacts:

```text
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_figure5_edit_strength_pareto.png
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_tshirt_star_seed10.png
experiments/support_v3_2026-06-02/e4_t1_t4_edit_strength_metrics.csv
experiments/support_v3_2026-06-02/table2a_e4_common_subset_clipdino_metrics.csv
```

Do not make the feedback claim from a single fixed-vs-full comparison. The
figure should show either a Pareto frontier or timestep trajectories.

## Figure 6 Limitation / Failure Figure

Keep failures tied to the current claim boundary:

```text
dog_remove_tennis_ball: occluded-object removal / host completion failure
whiteboard_remove_yellow_letter: glyph-field hallucination under blank removal
dog_replace_tennis_ball_star: partial replacement with mixed target/source residual
fridge magnet removals: accurate support but cluttered-surface completion damage
```

Do not reuse the old backpack-blue/yellow-car/rabbit-sunglasses panels as
current main-result evidence unless they are explicitly labeled as legacy
diagnostics.
