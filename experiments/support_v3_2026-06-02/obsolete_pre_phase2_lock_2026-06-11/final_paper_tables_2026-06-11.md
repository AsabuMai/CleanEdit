# Final Paper Tables (2026-06-11)

These tables are built from frozen CSV artifacts and the corrected T5 full-pillow task set.

Audit status: `complete` (`experiments/support_v3_2026-06-02/final_tables_audit_2026-06-11.json`).

## Table 1: Strict Core-5 Main Effect

| label | n | task_success_true | task_success_denominator | outside_l1 | inside_l1 | source_ssim_luma | dino_source | edit_score | clip_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF reconstruction | 15 | 0 | 9 | 0.0708 | 0.0564 | 0.5786 | 0.6153 | -0.0185 | -0.0185 |
| Direct target | 15 | 0 | 9 | 0.0911 | 0.0912 | 0.5141 | 0.5659 | -0.0042 | -0.0042 |
| Generic support | 15 | 1 | 9 | 0.0470 | 0.0791 | 0.6757 | 0.8798 | 0.0069 | 0.0069 |
| DeCE-RF-SD3 | 15 | 9 | 9 | 0.0393 | 0.0972 | 0.7094 | 0.8661 | 0.0476 | 0.0476 |

Success columns apply to relation-style T1/T2/T3 checks only; T4/T5 use operation-specific metrics and visual/material gates.

## Table 2a: Same-Backbone SD3 Common Subset

| label | n | outside_l1 | inside_l1 | source_ssim_luma | dino_source | edit_score | clip_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Direct target | 45 | 0.0729 | 0.1195 | 0.6318 | 0.6077 | 0.0096 | 0.0096 |
| Generic support | 45 | 0.0351 | 0.1111 | 0.7470 | 0.8998 | 0.0026 | 0.0026 |
| FlowEdit-SD3 | 45 | 0.1614 | 0.2026 | 0.4931 | 0.4873 | 0.0434 | 0.0434 |
| FlowAlign-SD3 | 45 | 0.0787 | 0.1339 | 0.7907 | 0.5769 | 0.0443 | 0.0443 |
| SplitFlow-SD3 | 45 | 0.0774 | 0.1422 | 0.7287 | 0.6284 | 0.0387 | 0.0387 |
| Fixed DeCE-SD3 | 45 | 0.0285 | 0.1440 | 0.9251 | 0.8619 | 0.0382 | 0.0382 |
| DeCE-RF-SD3 | 45 | 0.0279 | 0.1468 | 0.9254 | 0.8842 | 0.0333 | 0.0333 |

Table 2a is a same-protocol SD3/common-subset comparison. Fixed DeCE-SD3 is a component-control row, not an external baseline.

## Table 2b: Native Preservation-Aware RF / FLUX Context

| label | n | outside_l1 | inside_l1 | source_ssim_luma | dino_source | edit_score | clip_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FireFlow-FLUX/context | 45 | 0.0660 | 0.0991 | 0.7946 | 0.6806 | 0.0223 | 0.0223 |
| RF-Solver-Edit-FLUX/context | 45 | 0.0540 | 0.0806 | 0.8319 | 0.7505 | 0.0295 | 0.0295 |
| ReFlex-FLUX/context | 45 | 0.0830 | 0.1321 | 0.7913 | 0.7156 | 0.0329 | 0.0329 |

Table 2b is contextual and cross-backbone; do not use it for algorithm-level superiority claims.

## Table 3: Phase2 15-Task Breadth Validation

| label | n | outside_l1 | inside_l1 | source_ssim_luma | dino_source | edit_score | clip_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF reconstruction | 45 | 0.0487 | 0.0514 | 0.8528 |  |  |  |
| Direct target | 45 | 0.0729 | 0.1195 | 0.6318 | 0.6077 | 0.0096 | 0.0096 |
| Generic support | 45 | 0.0351 | 0.1111 | 0.7470 | 0.8998 | 0.0026 | 0.0026 |
| DeCE-RF-SD3 | 45 | 0.0279 | 0.1468 | 0.9254 | 0.8842 | 0.0333 | 0.0333 |

Phase2 is breadth/robustness validation over 15 controlled tasks, not a large-scale benchmark. E5 removal probes are excluded. Edit-score completeness is required for editing rows; the RF reconstruction row is a reconstruction floor.

## Table 3 Family Breakdown

| family | label | n | outside_l1 | inside_l1 | source_ssim_luma | dino_source | edit_score | clip_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_attached_accessory | RF reconstruction | 9 | 0.0848 | 0.0843 | 0.8204 |  |  |  |
| T1_attached_accessory | Direct target | 9 | 0.0947 | 0.1364 | 0.4277 | 0.5683 | -0.0034 | -0.0034 |
| T1_attached_accessory | Generic support | 9 | 0.0582 | 0.0490 | 0.5339 | 0.8935 | -0.0190 | -0.0190 |
| T1_attached_accessory | DeCE-RF-SD3 | 9 | 0.0568 | 0.1353 | 0.8715 | 0.9248 | 0.0718 | 0.0718 |
| T2_container_insertion | RF reconstruction | 9 | 0.0484 | 0.0427 | 0.8837 |  |  |  |
| T2_container_insertion | Direct target | 9 | 0.0733 | 0.0769 | 0.5984 | 0.4896 | 0.0112 | 0.0112 |
| T2_container_insertion | Generic support | 9 | 0.0388 | 0.0779 | 0.7132 | 0.7830 | 0.0164 | 0.0164 |
| T2_container_insertion | DeCE-RF-SD3 | 9 | 0.0341 | 0.0907 | 0.9141 | 0.8278 | 0.0042 | 0.0042 |
| T3_surface_decal | RF reconstruction | 9 | 0.0256 | 0.0222 | 0.9365 |  |  |  |
| T3_surface_decal | Direct target | 9 | 0.0467 | 0.0830 | 0.8277 | 0.7506 | 0.0008 | 0.0008 |
| T3_surface_decal | Generic support | 9 | 0.0239 | 0.0195 | 0.8828 | 0.9567 | -0.0175 | -0.0175 |
| T3_surface_decal | DeCE-RF-SD3 | 9 | 0.0168 | 0.0960 | 0.8984 | 0.7682 | 0.0639 | 0.0639 |
| T4_local_recolor | RF reconstruction | 9 | 0.0427 | 0.0690 | 0.9025 |  |  |  |
| T4_local_recolor | Direct target | 9 | 0.0596 | 0.1290 | 0.7407 | 0.7330 | 0.0192 | 0.0192 |
| T4_local_recolor | Generic support | 9 | 0.0297 | 0.3855 | 0.8290 | 0.9220 | 0.0352 | 0.0352 |
| T4_local_recolor | DeCE-RF-SD3 | 9 | 0.0297 | 0.3868 | 0.9736 | 0.9173 | 0.0328 | 0.0328 |
| T5_same_color_material | RF reconstruction | 9 | 0.0422 | 0.0388 | 0.7208 | 0.6998 | 0.0065 | 0.0065 |
| T5_same_color_material | Direct target | 9 | 0.0904 | 0.1724 | 0.5645 | 0.4969 | 0.0201 | 0.0201 |
| T5_same_color_material | Generic support | 9 | 0.0249 | 0.0236 | 0.7761 | 0.9438 | -0.0021 | -0.0021 |
| T5_same_color_material | DeCE-RF-SD3 | 9 | 0.0024 | 0.0252 | 0.9694 | 0.9827 | -0.0065 | -0.0065 |
