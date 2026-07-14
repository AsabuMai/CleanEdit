# Table Plan

Current source of truth: `paper/wacv_experiment_design.md` for the revised
experiment design and claim boundary, `paper/results.md` for the synchronized
readout, and
`experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md` for the
audited paper-facing tables. Archived server results are diagnostic only.

## Numbering Lock

Use the following paper-facing hierarchy. Do not use "Table 2 breadth" for the
Phase 2 expansion.

```text
Table 1: Strict Core-5 main effect
Table 2a: same-backbone SD3 algorithm comparison
Table 2b: native preservation-aware RF / FLUX contextual summary
Table S1, or Table 2c if space allows: support-matched diagnostic
Table 3: Phase 2 15-task common-subset breadth validation
Figure 4: support geometry ablation
Figure 5: controller / stress Pareto curve
Figure 6: E5 boundary and failure probes
```

This keeps the roles separate: Table 1 is the main effect, Table 2 is baseline
fairness, Table 3 is breadth/robustness validation, and the numbered figures
carry mechanism and boundary evidence.

## Main Comparison

Rows:

```text
strict E1 headline: 5 task instances x 4 paper-facing methods x seeds
10,11,12 = 60 complete rows in e1_core5_strict_metrics.csv
```

Headline task instances:

```text
cat_crown
bowl_apple_inside
tshirt_star
red_chair_blue
pillow_same_color_cable_knit
```

E5 boundary probe, not aggregated into Table 1:

```text
backpack_remove_toy_charm
```

Mapping into the updated Core-5 taxonomy:

```text
T1 attached accessory: cat_crown canonical; dog_sunglasses diagnostic only
T2 container-constrained insertion: bowl_apple_inside canonical
T3 surface decal: tshirt_star canonical; mug_heart diagnostic only
T4 object-level recolor: red_chair_blue canonical
T5 localized same-color material replacement: pillow_same_color_cable_knit canonical; pillow_same_color_corduroy_panel and pillow_vertical_fabric_strip diagnostic only
E5 removal boundary probe: backpack_remove_toy_charm, reported separately
```

Paper-facing methods:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
CleanEdit
```

Current artifacts:

```text
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/final_tables_audit_2026-06-11.json
experiments/support_v3_2026-06-02/table1_core5_final.csv
experiments/support_v3_2026-06-02/e1_core5_strict_metrics.csv
experiments/support_v3_2026-06-02/e1_core5_strict_metrics.json
experiments/support_v3_2026-06-02/eval_masks/eval_masks_metadata.json
experiments/support_v3_2026-06-02/mask_sensitivity_core5_summary.md
experiments/support_v3_2026-06-02/consolidated_2026-06-10/t5_cable_knit_gate_3seed.png
experiments/support_v3_2026-06-02/t6_removal_aware_metrics.csv
```

Headline columns should be grouped by the question they answer rather than as a
flat metric list:

- edit correctness: task-family success, relation correctness, and CLIP delta
  only as an auxiliary score
- preservation: outside-mask L1/RMSE, outside LPIPS/DINO when available, and
  luma/source SSIM
- locality: outside change energy, leakage near the boundary, and
  inside/outside change ratio
- perceptual quality: artifact score, visual audit, and overall preference

Inside-mask L1/RMSE is an edit-magnitude or local-change proxy, not target
correctness. T4 recolor should rely on object-mask color distance, target hue or
blue-ratio shift, and outside color leakage. T5 same-color material replacement
should rely on crop-level material audit, texture/high-frequency evidence,
artifact score, and outside preservation; CLIP and L1 must be interpreted
conservatively for T5.

support_v3_fixed should not appear as a headline E1 main-table method. Its
stable paper identity is Fixed CleanEdit (same-backbone component control): in E2.2
it is the preservation-control row, and in E4 it is the fixed-displacement
component ablation.

## Table 1: Strict Core-5 Main Results

Purpose:

```text
show the main localized edit-preserve effect under fixed evaluation masks
```

Rows use paper-facing names in tables, with runner names kept in metadata:

```text
Source reconstruction / RF reconstruction = base_only
Direct target RF guidance = direct_target
Generic support control = adaptive_full_generic_support
CleanEdit = support_v3_controller_rmsgap
```

Required caption point:

```text
All preservation and locality metrics are computed using fixed per-task
evaluation masks shared by all methods. These masks are not produced by
CleanEdit and are not adjusted after observing outputs. The table reports a
controlled diagnostic Core-5 suite, not a large-scale benchmark. The success
column is relation-style for T1/T2/T3; recolor and material replacement are
reported with task-specific evidence.
```

## Table 2a: Same-Backbone SD3 Algorithm Comparison

This is the main E2 algorithmic table.

Locked artifacts:

```text
experiments/support_v3_2026-06-02/table2a_sd3_common_subset_final.csv
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/final_tables_audit_2026-06-11.json
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed10_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed11_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed12_grid.png
```

Rows:

```text
direct_target-SD3
FlowEdit-SD3
FlowAlign-SD3
SplitFlow-SD3
Fixed CleanEdit-SD3 (same-backbone component control)
CleanEdit-SD3
```

Current locked E2.2 readout covers the audited T1-T5 common subset:

```text
15 tasks x 7 Table 2a rows x 3 seeds = summarized as 45 rows per method
rows: direct_target, generic_support, FlowEdit, FlowAlign, SplitFlow,
Fixed CleanEdit-SD3, CleanEdit-SD3
```

Optional same-backbone row only if genuinely verified:

```text
OT-RF/OTIP-SD3 or RF-Edit-SD3
```

Required columns:

```text
method | backbone | input condition | support used for control | edit success |
preserve fidelity | leakage/locality | NFE/runtime | caveat
```

Caption requirement:

```text
Algorithm-level conclusions are drawn only from same-backbone SD3 rows under
the same source images, prompts, seeds, and fixed evaluation masks.
```

## Table 2b: Native Preservation-Aware RF / FLUX Contextual Summary

This table is contextual, not the algorithmic leaderboard.

Locked artifacts:

```text
experiments/support_v3_2026-06-02/table2b_native_context_final.csv
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/e2_native_flux_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_native_flux_fixed_mask_metrics_with_context.csv
experiments/support_v3_2026-06-02/e2_native_flux_contextual_table.csv
experiments/support_v3_2026-06-02/e2_native_flux_contextual_table.md
experiments/support_v3_2026-06-02/e2_native_flux_normalized_512_manifest.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_visual_audit_filled.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_visual_audit_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_contextual_conclusion.md
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_grids/
```

Main-paper rows should include only runnable native/context baselines, typically
one to three rows. Planned or blocked rows belong in a supplement implementation
status table, not in a large main-paper table.

Runnable native/context block:

```text
RF-Solver-Edit / RF-Edit
ReFlex
FireFlow
```

Supplement implementation-status block:

```text
stable-flow, if the strict adapter is incomplete
OT-RF / OTIP-style, smoke verified but strict matrix pending
DVRF / Delta Velocity RF, smoke verified but strict matrix pending
```

Rows that remain blocked by access, repo, environment, or adapter gaps should be
documented in supplement with exact failure reasons. Do not silently drop them
and do not replace them with unrelated non-RF baselines.

Main-paper wording:

```text
We additionally report runnable native preservation-aware RF editors as
contextual baselines. Methods that could not be reliably adapted to our
fixed-mask protocol are documented in the supplement.
```

Caption requirement:

```text
Backbones and input conditions differ across native rows. This table evaluates
whether off-the-shelf preservation-aware RF editors solve the localized
edit-preserve tasks in practice; algorithm-level conclusions are drawn from the
same-backbone SD3 comparison.
```

## Table S1: Support-Matched Diagnostic

If main-paper space allows, this can be promoted to Table 2c. Otherwise it
should remain Supplement Table S1 with a short summary in the main text.

This is an oracle binary-localization diagnostic, not an ordinary baseline
comparison. It answers whether CleanEdit is only winning because it has access to
an edit mask/support region.

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e2_support_matched_diagnostic_manifest.csv
experiments/support_v3_2026-06-02/e2_support_matched_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.csv
experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.md
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_filled.csv
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_conclusion.md
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_grids/
```

Locked diagnostic rows:

```text
direct_target_raw
direct_target_mask_blend
flowedit_mask_blend
support_v3_controller_rmsgap
```

The 2026-06-05 design treats these as useful but not sufficient by themselves.
Current strengthened E2.4 status:

```text
direct_target_medit_gate: inference-time same fixed M_edit, 36/36 complete
FlowEdit + same M_edit gating: blocked; no verified mask-aware wrapper
Fixed CleanEdit / CleanEdit: component and full-method references, not support-only rows
```

Artifacts:

```text
experiments/support_v3_2026-06-02/e2_support_matched_medit_gate_metrics.csv
experiments/support_v3_2026-06-02/e2_support_matched_medit_gate_summary.csv
scripts/prepare_e2_support_matched_medit_gate.py
scripts/run_e2_support_matched_medit_gate_gpu01.sh
```

Interpretation:

```text
Binary localization improves outside preservation but does not reliably recover
relation correctness, boundary consistency, or target integration, indicating
that CleanEdit's gain is not explained by mask access alone.
```

Use only a binary edit support for baseline diagnostic rows. Do not give
baseline rows CleanEdit's `M_core`, `M_contact`, `M_preserve`, feedback weights,
or projection. Output blending and `M_edit` gating must be labeled as
diagnostic, not as fair natural-use baselines.

## Figure 4 / Supplement Table S2: Support Geometry Ablation

Purpose:

```text
show that support geometry is an explicit experimental object, not a hidden
hand-picked mask
```

Boundary:

```text
E3 fixes the controller and changes support geometry.
E4 fixes support and changes controller or stress axis.
```

Current locked scope:

```text
3 tasks x 6 support geometry variants x 3 seeds = 54 support-map rows
tasks: cat_crown, tshirt_star, backpack_remove_toy_charm
variants: attention_only, clean_disagreement, velocity_disagreement,
grounding_sam, generic_support, operation_conditioned_support
```

The 2026-06-05 design recommends a downstream runnable compact ablation when
compute allows:

```text
3 representative tasks x 6 support variants x 2 seeds = 36 outputs
support-ablation breadth: 5 headline categories x 2 examples x 6 variants x 2 seeds = 120 outputs
```

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_mask_metrics.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_summary.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_by_task_summary.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_correlation.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_summary.md
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_tshirt_star_seed10_figure4_panel.png
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_seed10_task_sheet.png
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_complete_2026-06-04.md
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_complete_2026-06-04.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry.sha256
```

Row interpretation:

```text
attention_only, clean_disagreement, velocity_disagreement, grounding_sam =
support-map diagnostics only

generic_support, operation_conditioned_support =
runnable downstream rows with support-quality and edit/preserve metrics
```

Do not report only support-map IoU or mask visualizations. The figure/table
must also report downstream edit behavior after feeding each support variant
into the same controller:

```text
edit success
relation correctness
outside drift
boundary artifact
locality / leakage
support area ratio
```

Paper-safe claim:

```text
Operation-conditioned support improves fixed-mask overlap and downstream edit
behavior relative to weak generic support, while Grounding/SAM alone tends to
over-cover the object/host region. This supports the claim that CleanEdit's
support geometry is not merely generic segmentation or raw attention evidence.
```

## Figure 5 / Supplement Table S3: Controller And Robustness Ablation

This section includes the component anchor and the final E4 controller/stress
package.

Boundary:

```text
E4 fixes operation-conditioned support and compares Fixed CleanEdit, CleanEdit full,
and selected controller/stress variants. Support perturbation is interpreted as
controller robustness stress, not as a new support-geometry comparison.
```

Component anchor artifacts:

```text
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact.csv
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact_rows.csv
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact.md
```

Current locked controller/stress scope:

```text
tasks: cat_crown, tshirt_star, pillow_same_color_cable_knit
base rows: support_v3_fixed, support_v3_controller_rmsgap x seeds 10/11/12 = 18
stress rows: support_v3_fixed, support_v3_controller_rmsgap x edit multipliers
0.50/0.75/1.00/1.25/1.50/2.00 x seed10 x 3 tasks = 36 metric rows
Phase2 T1-T4 expansion: 72/72 base rows, 144/144 edit-strength rows,
72/72 CLIP+DINO join rows
```

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e4_controller_base_metrics.csv
experiments/support_v3_2026-06-02/e4_edit_strength_metrics.csv
experiments/support_v3_2026-06-02/e4_t1_t4_controller_base_metrics.csv
experiments/support_v3_2026-06-02/e4_t1_t4_edit_strength_metrics.csv
experiments/support_v3_2026-06-02/table2a_e4_common_subset_clipdino_metrics.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_base_summary.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_edit_strength_summary.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_stats.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_summary.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_figure5_edit_strength_pareto.png
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_tshirt_star_seed10.png
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation_summary.md
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation_complete_2026-06-04.md
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation_complete_2026-06-04.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation.sha256
```

Paper-facing identity:

```text
Fixed CleanEdit (same-backbone component control) = support_v3_fixed
CleanEdit = support_v3_controller_rmsgap
```

Interpretation:

```text
Fixed CleanEdit displacement is a component ablation. It is not an external baseline
and not an E2.4 support-only row.
```

Paper-safe claim:

```text
E4 evaluates the full adaptive controller relative to a fixed-displacement CleanEdit
variant under the same SD3 implementation and fixed evaluation masks. The
stress curve should be reported as an edit-preserve tradeoff using local edit
L1 as an edit-pressure proxy, not as a standalone semantic success score.
```

Do not claim that feedback alone causes the improvement unless a separate
no-feedback row is present. If compute allows, put the following finer ablation
in supplement:

```text
Fixed CleanEdit
CleanEdit without feedback
CleanEdit without projection / clipping
Full CleanEdit
```

The revised design treats E4 as Pareto/stress evidence rather than a single
fixed-vs-feedback mean. Recommended Phase 2 stress axes:

```text
edit strength / guidance scale sweep
support perturbation sweep: erode, dilate, shift, noisy support
feedback/projection lambda or update-frequency sweep
```

Only write a stronger controller claim if the Pareto curve shows lower outside
drift at matched edit success, or higher edit success under a matched preserve
budget.

## Figure 6 / Supplement Table S4: Boundary And Failure Probes

Purpose:

```text
document scope boundary: where the base method stops working and which
extension routes are separate from the base CleanEdit mean
```

Scope:

```text
selected outputs: 36/36 complete
positive extension routes: high-confidence completion prior; replacement target route
failure labels: semantic glyph hallucination; cluttered-surface damage;
removal completion failure; replacement ambiguity
```

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_selected_manifest.csv
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_failure_taxonomy.csv
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_boundary_extension_summary.md
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_figure6_boundary_extension_seed10.png
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_gated_completion_protocol.json
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_whiteboard_red_star_protocol.json
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_boundary_extension_complete_2026-06-04.md
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_boundary_extension.sha256
```

Paper-safe claim:

```text
E5 documents extension routes and scope boundaries. It supports Figure 6 and
the limitations paragraph, but the extension routes are named separately and
are not aggregated into the base CleanEdit mean.
```

E5 should explicitly communicate that CleanEdit is not a general inpainting or
arbitrary replacement method. It is designed for localized edit-preserve control
when the desired local displacement is well defined.

## Same-Support Removal Diagnostic

A removal-only diagnostic was generated for `backpack_remove_toy_charm` using
Telea and Navier-Stokes OpenCV inpainting with the CleanEdit support mask. Report
it separately from the main comparison because it receives the same support mask
and only applies to removal/fill cases.

Artifact status: this is a legacy diagnostic row. The active repository no
longer tracks summary CSV/JSON files for this run; generated images remain
under ignored generated same-support inpaint output directories and should not
be cited as active paper evidence unless regenerated and summarized.

Readout: same-support inpainting gives lower outside drift on the backpack case
but produces visible fill artifacts around the strap/zipper region, while
CleanEdit removes the target charm but locally smooths the occluded zipper/fabric.

## Legacy Baseline Artifacts

Legacy baseline artifacts are historical audit material only:

```text
experiments/archive_legacy_2026-05-11/baseline_parity_manifest.csv
experiments/archive_legacy_2026-05-11/baseline_summary.csv
experiments/archive_legacy_2026-05-11/baseline_summary.md
paper/archive_old_core6_20260602/old_stage2_5_integrity_precheck.md
```

Do not use legacy Core-4 or old Core-6 artifacts as active paper evidence unless
they are rerun under the current strict protocol.

## Supplement: Audit And Mask Sensitivity

The 2026-06-05 design requires stronger evaluation transparency.

Blind internal audit:

```text
3 raters
randomized method order
source and target instruction visible
method name hidden
1-5 ratings for edit_correct, relation_correct, source_preservation, locality,
artifact_severity, overall, and controlled failure_type labels
```

Mask sensitivity:

```text
eroded evaluation mask
base evaluation mask
dilated evaluation mask
```

Operational mask-freeze protocol:

```text
define intended edit region from source image + target instruction
create base eval mask before inspecting method outputs
record mask metadata/hash
derive eroded and dilated variants
evaluate all methods/seeds with the same mask set
```

Main paper should report whether rankings are stable under mask sensitivity.
Full numbers belong in supplement. If preservation ranking is highly sensitive
to mask boundary choice, outside-mask metrics should be interpreted together
with visual audit rather than as standalone strong evidence.

## Table 3: Phase 2 15-Task Breadth Validation

The strict Core-5 canonical set supports a controlled main diagnostic. Phase 2
is breadth/robustness validation, not a large-scale, comprehensive, or general
image-editing benchmark. Use this main-text positioning:

```text
We further evaluate a 15-task common subset covering the same five operation
families to test whether the Core-5 trend remains stable across additional
sources.
```

The current Phase 2 target is:

```text
5 headline categories x 3 source examples x 4 E1 methods x 3 seeds = 180 internal E1 outputs
15 tasks x same-backbone/native baseline rows x seeds 10/11/12 for Table 3 breadth validation
E5 removal probes should be expanded and reported separately
```

Current locked artifacts:

```text
experiments/support_v3_2026-06-02/table3_phase2_breadth_internal_final.csv
experiments/support_v3_2026-06-02/table3_phase2_breadth_by_family_final.csv
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/final_tables_audit_2026-06-11.json
```

Supplement/reviewer-defense artifacts:

```text
experiments/support_v3_2026-06-02/mask_sensitivity_core5_summary.md
experiments/support_v3_2026-06-02/mask_sensitivity_core5_audit.json
experiments/support_v3_2026-06-02/efficiency_context_2026-06-11.md
experiments/support_v3_2026-06-02/efficiency_context_2026-06-11_audit.json
```

Earlier planning budgets remain useful for supplement/robustness, but they must
not be described as a large-scale benchmark:

```text
minimum: 5 headline categories x 2 source examples x 4 E1 methods x 3 seeds = 120 outputs
preferred: 5 headline categories x 3 source examples x 4 E1 methods x 3 seeds = 180 outputs
E5 removal probes should be expanded and reported separately
```

Priority is to add new source images for already-supported operations rather
than new relations:

```text
second accessory insertion
second surface decal
second local recolor
second exposed small-object removal
second inside-container insertion, after checking container-interior support
second material/surface-strip example
```

Aggregation rule:

```text
For each task, average seeds 10/11/12 first.
For Core-5, macro-average over the 5 tasks.
For Phase 2, report both overall macro-average over 15 tasks and per-family
macro-averages over the five operation families.
```

If space allows, report mean +/- standard error in the main table and bootstrap
confidence intervals in supplement.
