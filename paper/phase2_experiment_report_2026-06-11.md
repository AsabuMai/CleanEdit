# Phase2 Experiment Report 2026-06-11

## 1. Experiment Purpose

本阶段实验验证的核心主张不是“编辑幅度最大”，而是：

```text
DeCE-RF prioritizes non-edit-region preservation in localized RF editing while retaining measurable target-local edit signal.
```

中文表述：

```text
DeCE-RF 的目标是局部编辑时尽量保护非编辑区，同时保留可测的目标局部编辑信号。
```

因此，实验解释应以非编辑区保护为主线，避免写成通用图像编辑 SOTA、最大编辑强度、或全面自动编辑能力。

## 2. Locked Experiment Scope

当前项目口径已锁定为 Phase2：

```text
Phase2 = T1-T5, three source cases per family, seeds 10/11/12.
```

Phase2 是当前主实验阶段，不是单独的任务类别，也不是旧的 five-case Core-5。

| family | current cases |
| --- | --- |
| T1 attached accessory | `cat_crown`; `dog_bow_tie_phase2`; `dog_front_sunglasses_phase2` |
| T2 container insertion | `bowl_apple_inside`; `white_bowl_orange_tabletop_phase2`; `brown_bowl_lemon_phase2` |
| T3 surface decal | `tshirt_star`; `mug_heart`; `tote_leaf` |
| T4 local recolor | `red_office_chair_to_blue_office_chair`; `green_mug_orange_phase2`; `yellow_vase_blue_phase2` |
| T5 same-color material | `pillow_same_color_cable_knit`; `pillow_same_color_cable_knit_grey`; `pillow_same_color_cable_knit_armchair` |

旧 `red_chair_blue` 不属于当前 Phase2 evidence，只能作为归档历史材料。

## 3. Compared Methods

### Table 1 Main Internal Methods

| method | role in design |
| --- | --- |
| RF reconstruction | preservation floor / no-target-edit reference |
| Direct target | aggressive target guidance baseline |
| Generic support | conservative support-control baseline |
| DeCE-RF-SD3 | proposed preservation-first localized RF control |

### Table 2a Same-Backbone SD3 Context

Table 2a 在同一 SD3 backbone 和同一 15-case subset 上比较：

- Direct target
- Generic support
- FlowEdit-SD3
- SplitFlow-SD3
- Sam-Flow-SD3
- Fixed DeCE-SD3
- DeCE-RF-SD3

### Table 2b Native RF / FLUX Context

Table 2b 报告 preservation-aware native RF/FLUX context：

- FireFlow-FLUX/context
- RF-Solver-Edit-FLUX/context
- ReFlex-FLUX/context
- Sam-Flow-FLUX/context

这部分应作为 context，不应写成严格 same-backbone 直接胜负。

## 4. Metrics And Interpretation

| metric | direction | interpretation |
| --- | --- | --- |
| `outside_l1` / `outside_mask_l1` | lower is better | 非编辑区像素变化越小，保护越好 |
| `source_ssim_luma` | higher is better | 源图结构和亮度一致性越高越好 |
| `dino_source` | higher is better | 源图语义身份保持越好 |
| `edit_score` / `clip_delta` | generally higher is stronger edit signal | 目标局部编辑信号；不能单独代表整体质量 |
| `inside_l1` | descriptive | 编辑区变化强度，不是单调越高越好 |

### Final Selected Run Registry

Final qualitative figures are resolved through:

```text
experiments/support_v3_2026-06-02/phase2_final_selected_runs_2026-06-11.csv
```

The registry has 540 rows for T1-T5 x seeds 10/11/12 x formal methods,
including the retained Sam-Flow-SD3 and Sam-Flow-FLUX/context baselines. It
records the selected run, selected image, selection reason, and
metric-alignment status.

Current status is 540 `aligned` rows and 0 `needs_metric_refresh` rows. The
2026-06-12 selected-run metric refresh re-evaluated the 33 repaired/formal
DeCE-RF visual selections and merged them into `phase2_internal_bg_metrics.csv`.

实验主张要求同时看 preservation 与 edit signal。仅有低 outside L1 可能只是 no-op；仅有高 edit signal 可能破坏非编辑区。

## 5. Main Qualitative Figure

主图使用每个任务族一个代表 seed-12 case：

```text
source | direct target | generic support | DeCE-RF
```

文件：

```text
paper/assets/phase2_main_qual_grid_seed12_2026-06-11.png
paper/assets/phase2_main_qual_grid_seed12_2026-06-11.json
```

![Phase2 main qualitative grid](assets/phase2_main_qual_grid_seed12_2026-06-11.png)

图中可见的实验逻辑：

- Direct target 往往更激进，但容易改变源图布局、背景或身份。
- Generic support 更保守，非编辑区较稳，但容易 under-edit。
- DeCE-RF 的定位是局部编辑与非编辑区保持之间的折中，特别强调非编辑区保护。

## 6. Table 1: Phase2 T1-T5 Main Effect

Scope:

```text
15 task cases x seeds 10/11/12 = 45 rows per method
```

| label | n | outside_l1 | inside_l1 | source_ssim_luma | dino_source | edit_score |
| --- | --- | --- | --- | --- | --- | --- |
| RF reconstruction | 45 | 0.0487 | 0.0514 | 0.8528 |  |  |
| Direct target | 45 | 0.0729 | 0.1195 | 0.6318 | 0.6077 | 0.0096 |
| Generic support | 45 | 0.0351 | 0.1111 | 0.7470 | 0.8998 | 0.0026 |
| DeCE-RF-SD3 | 45 | 0.0279 | 0.1468 | 0.9254 | 0.8842 | 0.0333 |

Metric alignment: Table 1 is generated from the current metric CSVs after the
selected-run refresh. The final selected run registry reports 540/540 aligned
rows.

### Table 1 Interpretation

Table 1 支持 preservation-first 主张：

- DeCE-RF-SD3 的 outside L1 最低，说明非编辑区变化最小。
- DeCE-RF-SD3 的 source SSIM luma 最高，说明源图结构保持最好。
- DeCE-RF-SD3 的 edit_score 为正，说明不是单纯 no-op。
- Direct target 有目标压力，但 outside L1 和 source consistency 明显更差。
- Generic support 的 preservation 尚可，但 edit_score 接近 0，说明容易 under-edit。
- RF reconstruction 是 preservation floor，但不提供目标编辑信号。

结论：DeCE-RF-SD3 最符合“保护非编辑区，同时保留局部编辑信号”的设计目标。

## 7. Table 2a: Same-Backbone SD3 Context

| label | n | outside_l1 | inside_l1 | source_ssim_luma | dino_source | edit_score |
| --- | --- | --- | --- | --- | --- | --- |
| Direct target | 45 | 0.0729 | 0.1195 | 0.6318 | 0.6077 | 0.0096 |
| Generic support | 45 | 0.0351 | 0.1111 | 0.7470 | 0.8998 | 0.0026 |
| FlowEdit-SD3 | 45 | 0.1614 | 0.2026 | 0.4931 | 0.4873 | 0.0434 |
| SplitFlow-SD3 | 45 | 0.0774 | 0.1422 | 0.7287 | 0.6284 | 0.0387 |
| Sam-Flow-SD3 | 45 | see `phase2_paper_tables_2026-06-11.md` | see current table | see current table | see current table | see current table |
| Fixed DeCE-SD3 | 45 | 0.0285 | 0.1440 | 0.9251 | 0.8619 | 0.0382 |
| DeCE-RF-SD3 | 45 | 0.0279 | 0.1468 | 0.9254 | 0.8842 | 0.0333 |

### Table 2a Interpretation

Table 2a 说明 DeCE-RF 的优势不是最大编辑强度，而是 preservation-first balance：

- FlowEdit-SD3 和 SplitFlow-SD3 更偏编辑激进，但 non-edit preservation 更弱；Sam-Flow-SD3 是保留的最近外部 SD3 baseline。
- Fixed DeCE-SD3 与 DeCE-RF-SD3 非常接近，说明 DeCE clean-estimate decomposition 本身是主要贡献。
- DeCE-RF-SD3 相比 Fixed DeCE-SD3 preservation 略好，但不能强写 adaptive feedback 带来巨大独立提升。

推荐写法：

```text
The adaptive rmsgap path stabilizes the preservation-oriented DeCE behavior,
but the main empirical contribution is the clean-estimate edit-preserve
decomposition rather than a large standalone feedback gain.
```

## 8. Table 2b: Native RF / FLUX Context

| label | n | outside_l1 | inside_l1 | source_ssim_luma | dino_source | edit_score |
| --- | --- | --- | --- | --- | --- | --- |
| FireFlow-FLUX/context | 45 | 0.0660 | 0.0991 | 0.7946 | 0.6806 | 0.0223 |
| RF-Solver-Edit-FLUX/context | 45 | 0.0540 | 0.0806 | 0.8319 | 0.7505 | 0.0295 |
| ReFlex-FLUX/context | 45 | 0.0830 | 0.1321 | 0.7913 | 0.7156 | 0.0329 |
| Sam-Flow-FLUX/context | 45 | see `phase2_paper_tables_2026-06-11.md` | see current table | see current table | see current table | see current table |

### Table 2b Interpretation

Native RF/FLUX rows provide external context. They show that preservation-aware native methods can keep moderate edit signal, but their non-edit-region preservation is weaker than DeCE-RF-SD3 in the SD3 same-subset table. Because backbone and implementation context differ, this should be presented cautiously as context rather than direct SOTA ranking.

## 9. Family-Level Notes

Family-level breakdown is stored in:

```text
experiments/support_v3_2026-06-02/phase2_t1_t5_family_breakdown_final.csv
```

Important patterns:

- T1 attached accessory and T3 surface decal show clear DeCE-RF target-local edit signal with better preservation than direct target.
- T4 recolor is a strong example of preservation: DeCE-RF and generic support have similar outside L1, but DeCE-RF source SSIM is much higher.
- T5 same-color material has excellent preservation, but automatic edit metrics are more delicate because same-color texture changes are hard for CLIP-style scores. Discuss T5 cautiously.

## 10. Proxy Blind Internal Audit

Current package:

```text
experiments/support_v3_2026-06-02/blind_internal_audit_phase2_t1_t5_2026-06-11/
```

Status:

- 3 raters
- 45 items per rater
- 180 rows per rater
- 540 total manifest rows
- missing images = 0
- score audit = complete

The current sheets are filled with Codex metric-guided proxy ratings, not human ratings. This must be described as an internal proxy precheck only.

Proxy summary:

| method_display | n | edit_correct | relation_correct | source_preservation | locality | artifact_severity | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Generic support control | 135 | 2.044 | 2.044 | 4.356 | 4.533 | 2.067 | 3.526 |
| RF reconstruction / base reconstruction | 135 | 1.000 | 1.000 | 4.133 | 4.267 | 2.067 | 2.844 |
| Direct target guidance | 135 | 1.756 | 1.756 | 1.244 | 1.711 | 4.593 | 1.489 |
| DeCE-RF | 135 | 4.022 | 4.022 | 4.756 | 4.667 | 1.667 | 4.667 |

Proxy audit aligns with the preservation-first quantitative result, but it cannot be used as human-study evidence.

## 11. Claim Validation

### Supported

当前实验支持以下主张：

```text
On a controlled Phase2 T1-T5 diagnostic set, DeCE-RF improves non-edit-region
preservation and source consistency while retaining measurable target-local
edit signal.
```

证据链：

- Table 1: DeCE-RF-SD3 has best outside L1 and source SSIM among internal methods.
- Table 2a: same-backbone context shows stronger preservation than edit-aggressive SD3 baselines.
- Main qualitative grid: visually shows direct-target drift and DeCE-RF preservation behavior.
- Proxy audit: internal sanity check agrees with the preservation-first trend,
  but it is not human evidence.

Caveat: the claim is supported as a Phase2 closure direction. The selected-run
registry and metric CSVs are now aligned, but the manuscript should still avoid
human-study wording unless real human ratings replace the proxy-filled sheets.

### Not Supported

当前实验不支持以下过强主张：

- DeCE-RF is a broad general-purpose image editor.
- DeCE-RF is universally better than every baseline under every metric.
- DeCE-RF has the strongest edit amplitude.
- Adaptive feedback alone is the decisive source of improvement.
- Same-color material editing is completely solved.
- Proxy-filled audit scores count as human evidence.

## 12. Current Decisions

### Phase3

Do not enter Phase3 yet. Phase2 already supports the conservative preservation-first paper claim. The immediate task is Phase2 paper closure.

### Human Blind Audit

Human blind audit is optional for Phase2 closure. It becomes necessary only if the paper wants to make human visual-audit claims or if an advisor/reviewer asks for independent visual judgment.

Decision file:

```text
docs/human_blind_audit_decision_2026-06-11.md
```

### Phase2 Mask Sensitivity

Do not run immediately. Run only if the final paper needs explicit mask-robustness defense, if human ratings conflict with metrics, or if a reviewer/advisor asks.

Decision file:

```text
docs/phase2_mask_sensitivity_decision_2026-06-11.md
```

## 13. Final Reporting Recommendation

Recommended paper wording:

```text
DeCE-RF is designed primarily for non-edit-region preservation in localized RF
editing. On the Phase2 T1-T5 diagnostic set, it achieves the strongest
outside-mask preservation and source consistency among the main internal
methods while retaining measurable target-local edit signal.
```

Recommended Chinese summary:

```text
DeCE-RF 的核心目标是局部编辑中的非编辑区保护。在 Phase2 T1-T5 诊断集上，
它取得了最好的非编辑区保持和源图一致性，同时保留了可测的目标局部编辑信号。
```

## 14. Appendix Visual Comparisons

The appendix grids provide per-task visual comparisons for all formal main and external baseline methods:

```text
Source | RF recon | Direct | Generic | FlowEdit | SplitFlow | Sam-Flow-SD3 | FireFlow | RF-Solver | ReFlex | Sam-Flow-FLUX | Fixed DeCE | DeCE-RF
```

The grids use seed 12 for every Phase2 task case.

| figure | path |
| --- | --- |
| Figure A1: T1 attached accessory | `paper/assets/appendix_grids/appendix_T1_attached_accessory_representative_all_methods.png` |
| Figure A2: T2 container insertion | `paper/assets/appendix_grids/appendix_T2_container_insertion_representative_all_methods.png` |
| Figure A3: T3 surface decal | `paper/assets/appendix_grids/appendix_T3_surface_decal_representative_all_methods.png` |
| Figure A4: T4 local recolor | `paper/assets/appendix_grids/appendix_T4_local_recolor_representative_all_methods.png` |
| Figure A5: T5 same-color material | `paper/assets/appendix_grids/appendix_T5_same_color_material_representative_all_methods.png` |

## 15. Active Artifacts

| artifact | path |
| --- | --- |
| scope lock | `PHASE2_LOCK_2026-06-11.md` |
| current status | `CURRENT_PHASE2_STATUS_2026-06-11.md` |
| table summary | `experiments/support_v3_2026-06-02/phase2_paper_tables_2026-06-11.md` |
| table audit | `experiments/support_v3_2026-06-02/phase2_tables_audit_2026-06-11.json` |
| final selected run registry | `experiments/support_v3_2026-06-02/phase2_final_selected_runs_2026-06-11.csv` |
| main figure | `paper/assets/phase2_main_qual_grid_seed12_2026-06-11.png` |
| figure manifest | `paper/assets/phase2_main_qual_grid_seed12_2026-06-11.json` |
| proxy audit summary | `experiments/support_v3_2026-06-02/blind_internal_audit_phase2_t1_t5_2026-06-11/blind_internal_audit_summary.md` |
| human audit decision | `docs/human_blind_audit_decision_2026-06-11.md` |
| mask sensitivity decision | `docs/phase2_mask_sensitivity_decision_2026-06-11.md` |
