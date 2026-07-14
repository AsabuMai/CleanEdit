# Current Results

Date: evidence lock synchronized through 2026-06-11 P0 completion.

Scope: current WACV evidence package under the revised strict Core-5 headline
suite plus the E5 removal boundary probe. Older 2026-06-01 server-evidence
tables and pre-rescope Core-6 wording are archived under
paper/archive_old_core6_20260602/ and should be used only as supplementary
diagnostics.

Primary current evidence files:

```text
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/final_tables_audit_2026-06-11.json
experiments/support_v3_2026-06-02/table1_core5_final.csv
experiments/support_v3_2026-06-02/table2a_sd3_common_subset_final.csv
experiments/support_v3_2026-06-02/table2b_native_context_final.csv
experiments/support_v3_2026-06-02/table3_phase2_breadth_internal_final.csv
experiments/support_v3_2026-06-02/e1_core5_strict_metrics.csv
experiments/support_v3_2026-06-02/mask_sensitivity_core5_summary.md
experiments/support_v3_2026-06-02/efficiency_context_2026-06-11.md
experiments/support_v3_2026-06-02/t6_removal_aware_metrics.csv
```

Revised design source:

```text
paper/wacv_experiment_design.md
```

## Active Strict Core-5 Plus E5 Probe

The current headline E1 task set is:

```text
T1 attached accessory: cat_crown
T2 container-constrained insertion: bowl_apple_inside
T3 surface decal: tshirt_star
T4 local recolor: red_chair_blue
T5 localized same-color material replacement: pillow_same_color_cable_knit
```

Removal is no longer a headline E1 row:

```text
E5 boundary probe: backpack_remove_toy_charm
```

The diagnostic rows dog_sunglasses, mug_heart, pillow_blue_stripes,
pillow_vertical_fabric_strip, and pillow_same_color_corduroy_panel are not
canonical strict Core-5 rows. The old T6 removal row is reported only as a
boundary/completion probe because removal has no well-defined clean-displacement
edit target.

## Protocol

Paper-facing E1 methods:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
CleanEdit
```

Runner names:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
```

`support_v3_fixed` is retained as Fixed CleanEdit displacement. It should not appear
as a headline E1 main-table method, but the 2026-06-05 design uses it as an E4
component/controller ablation and as the E2.2 same-backbone SD3
preservation-control row when joined with the strict cache.

All preservation and locality metrics are computed using fixed per-task
evaluation masks shared by all methods. These masks are not produced by
CleanEdit and are not adjusted after observing outputs.

## Locked Evidence Package

E1 strict Core-5 metrics and visual/material gates:

```text
experiments/support_v3_2026-06-02/e1_core5_strict_metrics.csv
experiments/support_v3_2026-06-02/e1_core5_strict_metrics.json
experiments/support_v3_2026-06-02/task_success_validation_metrics.csv
experiments/support_v3_2026-06-02/consolidated_2026-06-10/t5_cable_knit_gate_3seed.png
experiments/support_v3_2026-06-02/eval_masks/
experiments/support_v3_2026-06-02/normalized_512/normalized_512_manifest.csv
```

E2.2 same-backbone SD3 algorithm comparison:

```text
experiments/support_v3_2026-06-02/table2a_sd3_common_subset_final.csv
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/final_tables_audit_2026-06-11.json
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
```

E2.3 native FLUX contextual comparison:

```text
experiments/support_v3_2026-06-02/table2b_native_context_final.csv
experiments/support_v3_2026-06-02/final_paper_tables_2026-06-11.md
experiments/support_v3_2026-06-02/e2_native_flux_normalized_512_manifest.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_visual_audit_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_contextual_conclusion.md
```

E2.4 support-matched diagnostic:

```text
experiments/support_v3_2026-06-02/e2_support_matched_diagnostic_manifest.csv
experiments/support_v3_2026-06-02/e2_support_matched_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.csv
experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.md
experiments/support_v3_2026-06-02/e2_support_matched_medit_gate_metrics.csv
experiments/support_v3_2026-06-02/e2_support_matched_medit_gate_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_conclusion.md
```

E3-E5 mechanism, controller, and boundary packages:

```text
experiments/support_v3_2026-06-02/e3_support_geometry/
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact.md
experiments/support_v3_2026-06-02/e4_controller_ablation/
experiments/support_v3_2026-06-02/e5_boundary_extension/
```

## Main Readout

The current strict Core-5 Table 1 matrix is complete and audited:

```text
5 tasks x 4 paper-facing methods x 3 seeds = 60 complete runs
audit: final_tables_audit_2026-06-11.json status=complete
```

Table 1 readout:

- CleanEdit: 9/9 relation-style success checks on T1/T2/T3, outside L1 0.0393,
  luma SSIM 0.7094, edit-score/CLIP delta 0.0476.
- Generic support: 1/9 relation-style success checks, outside L1 0.0470,
  luma SSIM 0.6757, edit-score/CLIP delta 0.0069.
- Direct target: 0/9 relation-style success checks, outside L1 0.0911,
  luma SSIM 0.5141, edit-score/CLIP delta -0.0042.
- RF reconstruction: reconstruction floor, outside L1 0.0708 and no edit.

The current canonical T5 is `pillow_same_color_cable_knit`. It should be
evaluated as localized same-color material replacement, not as a blue strip,
decal, recolor row, or old center-panel probe. The Phase 2 formal T5 set is
`pillow_same_color_cable_knit`, `pillow_same_color_cable_knit_grey`, and
`pillow_same_color_cable_knit_armchair`; `pillow_same_color_linen_panel` and
`pillow_same_color_terry_panel` remain diagnostic only.

T4 recolor and T5 material replacement should be discussed with
operation-specific visual/material gates rather than the T1/T2/T3 relation
success count.

Direct target guidance remains an aggressive baseline: it may produce target
semantics, but often changes source identity, crop/layout, or background.
Generic support control remains a strong preservation baseline, but often
over-preserves and misses the intended edit.

## E2 Readout

E2 is locked as a backbone-controlled and preservation-aware fairness
experiment, not a single RF-baseline leaderboard.

### E2.1 Protocol And Calibration Lock

E2.1 fixes the source set, fixed evaluation masks, normalized 512 display/eval
copies, strict readout, and human visual-audit protocol. SD3 reconstruction and
direct-target floors are taken from the strict E1 rows. Native FLUX rows are
reported with their own backbone/context columns rather than merged into a
same-backbone leaderboard.

### E2.2 Same-Backbone SD3 Evidence

Completed strict SD3 rows:

```text
Direct target-SD3
Generic support-SD3
FlowEdit-SD3
FlowAlign-SD3
SplitFlow-SD3
Fixed CleanEdit-SD3
CleanEdit-SD3
```

Table 2a uses the audited T1-T5 common subset: 45 rows per method. CleanEdit-SD3
has the lowest outside L1 among the Table 2a rows (0.0279) and the highest
luma SSIM (0.9254), while Fixed CleanEdit-SD3 is very close on preservation
(outside L1 0.0285, luma SSIM 0.9251). The feedback claim should therefore be
framed through the controller/stress evidence, not as a large single-row jump.
FlowEdit, FlowAlign, and SplitFlow remain the completed same-backbone SD3
RF-native baselines. This supports only the narrow same-backbone claim; it is
not a claim over all RF or FLUX editors.

### E2.3 Native FLUX Context

Native FLUX preservation-aware RF rows are now locked as contextual evidence.
They test whether off-the-shelf native implementations solve the same localized
edit-preserve setting in practice, but backbone and interface differences mean
they do not replace the same-backbone E2.2 algorithmic claim.

Main paper should report runnable native/context rows only, typically
RF-Solver-Edit / RF-Edit, ReFlex, and FireFlow. stable-flow remains
adapter/matrix-pending. OT-RF / OTIP and DVRF are repo/env/help-smoke and
single-case generation-smoke verified, but strict fixed-mask matrices are still
pending; they belong in a supplement implementation-status table until
promoted by a complete matrix.

Paper-facing interpretation:

```text
We additionally report runnable native preservation-aware RF editors as
contextual baselines. Methods that could not be reliably adapted to our
fixed-mask protocol are documented in the supplement.
```

### E2.4 Support-Matched Diagnostic

Locked rows:

```text
direct_target_raw
direct_target_mask_blend
flowedit_mask_blend
support_v3_controller_rmsgap
```

This is an oracle binary-localization diagnostic, not an ordinary baseline
comparison. It answers whether binary localization alone explains the CleanEdit
result. The locked readout is that binary localization improves outside
preservation but does not reliably recover relation correctness, boundary
consistency, or target integration. Therefore CleanEdit's gain is not explained by
mask access alone.

The 2026-06-11 update adds an inference-time `direct_target_medit_gate` row
using the same fixed edit masks over the Phase-2 T1-T4 subset: 36/36 complete,
outside L1 0.0574, inside L1 0.1923, and source SSIM-luma 0.6557. This is a
stronger diagnostic than post-hoc blending for direct target. FlowEdit remains
post-hoc-only because no verified mask-aware FlowEdit inference wrapper exists
in this project.

### Safe E2 Wording

Use:

```text
Under the same SD3 backbone, same prompts/source images, and fixed evaluation
masks, CleanEdit improves localized edit-preserve behavior over runnable RF-native
SD3 editing baselines in the reduced strict comparison.
```

Use for native rows:

```text
Native preservation-aware RF editors are reported as implementation-context
baselines because their public routes use different backbones or interfaces.
```

Do not write:

```text
broad all-RF victory claim.
direct FLUX superiority claim.
SD3-CleanEdit is directly superior to ReFlex-FLUX or RF-Edit-FLUX as an algorithm.
```

## Component And Mechanism Evidence

E3 treats the support mask as an explicit experimental object:

```text
3 tasks x 6 support geometry variants x 3 seeds = 54 support-map rows
tasks: cat_crown, tshirt_star, backpack_remove_toy_charm
variants: attention_only, clean_disagreement, velocity_disagreement,
grounding_sam, generic_support, operation_conditioned_support
```

Paper-safe E3 claim:

```text
Operation-conditioned support improves fixed-mask overlap and downstream edit
behavior relative to weak generic support, while Grounding/SAM alone tends to
over-cover the object/host region. Attention, clean-disagreement,
velocity-disagreement, and Grounding/SAM rows are support-map diagnostics, not
full editing baselines.
```

E3 should report both support-map quality and downstream behavior after feeding
each support variant into the same controller: edit success, relation
correctness, outside drift, boundary artifact, locality/leakage, and support
area ratio.

E4 is complete for the compact SD3 controller-ablation subset and has a
Phase2 T1-T4 expansion for Table 2a joining:

```text
tasks: cat_crown, tshirt_star, pillow_same_color_cable_knit
base rows: support_v3_fixed, support_v3_controller_rmsgap x seeds 10/11/12 = 18
stress rows: support_v3_fixed, support_v3_controller_rmsgap x edit multipliers
0.50/0.75/1.00/1.25/1.50/2.00 x seed10 x 3 tasks = 36 metric rows
T1-T4 expansion: 72/72 base rows, 144/144 edit-strength rows, 72/72 CLIP+DINO join rows
```

`support_v3_fixed` should be described as Fixed CleanEdit displacement. It isolates
decoupled clean-estimate displacement with operation-conditioned support but
without the feedback/projection controller.

Paper-safe E4 claim:

```text
E4 evaluates the full adaptive controller relative to a fixed-displacement CleanEdit
variant under the same SD3 implementation and fixed evaluation masks. The
stress curve should be reported as an edit-preserve tradeoff using local edit
L1 as an edit-pressure proxy, not as a standalone semantic success score. Do
not claim that feedback alone causes the improvement unless a separate
no-feedback row is present.
```

Mask sensitivity is complete for the Core-5 Table 1 rows. Under eroded, base,
and dilated fixed evaluation masks, the outside-L1 ranking remains stable:

```text
CleanEdit-SD3 > Generic support > RF reconstruction > Direct target
```

The base outside-L1 values are CleanEdit 0.0393, Generic support 0.0470, RF
reconstruction 0.0708, and Direct target 0.0911. Full values are in
`mask_sensitivity_core5_summary.md`.

## Phase 2 Breadth Readout

Phase 2 is Table 3 breadth/robustness validation, not a large-scale benchmark.
Use this framing:

```text
We further evaluate a 15-task common subset covering the same five operation
families to test whether the Core-5 trend remains stable across additional
sources.
```

Report task-level seed means first, then macro-average over tasks and operation
families. E5 removal probes remain boundary evidence and are not aggregated into
the T1-T5 breadth mean.

The audited Table 3 internal common subset has 45 rows per method. CleanEdit-SD3
has outside L1 0.0279, luma SSIM 0.9254, DINO source 0.8842, and edit-score
0.0333. Generic support has outside L1 0.0351 and higher DINO source 0.8998
but much lower edit-score 0.0026. The RF reconstruction row has blank
DINO/edit/CLIP fields in Table 3 unless those metrics are complete for all
tasks.

The efficiency/context table is complete for 11 methods x 45 Phase2 rows.
Internal SD3 rows record runtime and peak memory; external baselines disclose
runtime/peak as unavailable when the producing runner did not record them.
CleanEdit-SD3 uses the same `T_steps=28,n_max=24` setting as the internal SD3
rows, with mean runtime 56.89s and peak 13.35GB, so the main Table 3 trend
should not be described as purchased by a larger NFE budget.

## Boundary And Extension Evidence

E5 is complete for the selected boundary/extension package:

```text
selected outputs: 36/36 complete
positive extension routes: high-confidence completion prior; replacement target route
failure labels: semantic glyph hallucination; cluttered-surface damage;
removal completion failure; replacement ambiguity
```

Paper-safe E5 claim:

```text
E5 documents extension routes and scope boundaries. It supports Figure 6 and
the limitations paragraph, but the extension routes are named separately and
are not aggregated into the base CleanEdit mean.
```

## Claim Boundary

The current evidence supports:

```text
CleanEdit improves localized edit-preserve control under reasonable support
across insertion, surface editing, local recolor, and same-color material
replacement within a controlled Core-5 diagnostic suite. Simple exposed removal
is reported separately as E5 boundary evidence.
```

The current evidence does not support:

```text
broad arbitrary removal/replacement
occluded-object removal requiring substantial host completion
precise glyph replacement
state-of-the-art general-purpose image editing
cross-backbone CleanEdit transfer to FLUX
algorithm-level superiority over native FLUX editors
```

E2.5 cross-backbone CleanEdit transfer is deferred. All algorithm-level
conclusions should be drawn from same-backbone SD3 comparisons; native FLUX rows
are contextual evidence for off-the-shelf RF editors.

## Next Drafting Steps

1. Use the locked E1/E2/E3/E4/E5 artifacts as the paper evidence package.
2. Build main Table 1, Table 2a/2b, Table 3, and supplement S1/S2/S3/S4 tables from the
   locked artifact lists.
3. In Table 2a, include Fixed CleanEdit-SD3 as the same-backbone preservation-control
   row when the strict fixed cache is joined.
4. Update the experiment section around the evidence chain:
   main effect -> baseline fairness -> breadth validation -> mechanism ablation -> boundary cases.
5. Keep old server-evidence rows only as supplement/diagnostic material.
6. Keep extension routes out of the base CleanEdit aggregate unless the method
   column explicitly names the extra route.
7. Treat Phase 2 breadth, blind internal audit, erode/base/dilate mask
   sensitivity, and stronger E2.4 inference-time same-support diagnostics as
   supplement/robustness paths before making broader claims, not as prerequisites
   for the current controlled Core-5 readout.
