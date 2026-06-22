# Phase2 Table 3 / Table 2 Coverage Audit 2026-06-11

Scope note: Phase2 breadth coverage is T1-T5, 15 tasks total, with seeds
10/11/12. Its paper-facing home is Table 3, the 15-task common-subset
breadth/robustness validation. The same completed rows may also feed Table 2a
or Table 2b baseline summaries when the comparison is explicitly scoped by
backbone/context. E5 removal has 3 additional boundary-probe tasks and should
stay separate unless explicitly scoped.

## Phase2 Task Set

- T1 attached accessory: `cat_crown`, `dog_bow_tie_phase2`, `dog_front_sunglasses_phase2`
- T2 insertion: `bowl_apple_inside`, `white_bowl_orange_tabletop_phase2`, `brown_bowl_lemon_phase2`
- T3 surface decal: `tshirt_star`, `mug_heart`, `tote_leaf`
- T4 recolor: `red_office_chair_to_blue_office_chair`, `green_mug_orange_phase2`, `yellow_vase_blue_phase2`
- T5 same-color material replacement, formal T5-1-style full-pillow set:
  `pillow_same_color_cable_knit`, `pillow_same_color_cable_knit_grey`,
  `pillow_same_color_cable_knit_armchair`
- E5 removal boundary: `backpack_remove_toy_charm`, `backpack_remove_silver_keychain_phase2`, `bag_remove_decorative_tag_phase2`

## Current Coverage

- T1-T4 are complete for internal `pretty_matrix` methods: base, direct, generic, DeCE-RF, and Fixed DeCE, all 12 tasks x 3 seeds.
- T1-T4 are complete for baseline outputs/metrics: FlowEdit, FlowAlign, SplitFlow, FireFlow, RF-Solver-Edit, and ReFlex, all 12 tasks x 3 seeds.
- T1-T4 Fixed DeCE and DeCE-RF CLIP+DINO metrics were completed in `table2a_e4_common_subset_clipdino_metrics.csv`.
- T5 formal full-pillow cable-knit set is complete for the corrected task set:
  - Internal metrics: 45 rows = 3 tasks x 5 methods x 3 seeds in
    `experiments/support_v3_2026-06-02/table2_t5_internal_metrics.csv`.
  - Baseline metrics: 54 rows = 3 tasks x 6 baselines x 3 seeds in
    `experiments/support_v3_2026-06-02/table2_t5_baseline_metrics.csv`.
  - Baseline manifest and matrix are complete: 54/54 rows, no missing rows in
    `experiments/support_v3_2026-06-02/e2_t5_baseline_matrix_missing.csv`.
  - `pillow_same_color_cable_knit_grey` and
    `pillow_same_color_cable_knit_armchair` were filled by Slurm job 717149;
    corrected CLIP/DINO metrics were finalized by metrics-only job 717295.
  - The previously completed `pillow_same_color_linen_panel` and
    `pillow_same_color_terry_panel` rows are old center-panel T5 probes. They
    must stay diagnostic and must not be used as final Table 3 T5 rows.
- E5 removal is boundary evidence and not part of the headline T1-T5 Table 3 common subset.

## Table 3 / Table 2 Rule

Do not promote the older `table2a_sd3_common_subset.csv` as final. Rebuild the
final Table 3 breadth validation, and any Table 2a/Table 2b summaries derived
from Phase2, from the common task/seed subset that includes the corrected T5
full-pillow metrics above. Keep the old linen/terry panel metrics out of the
final paper-facing tables.
