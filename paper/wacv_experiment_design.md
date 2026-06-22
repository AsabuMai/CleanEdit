# WACV Experiment Design: FlowEdit-135 Main Evaluation

## Scope

Current main paper scope:

```text
FlowEdit-compatible 135 tasks, grouped into T1-T5 localized editing families.
```

This replaces the older Phase2 15-case diagnostic set as the main paper
evidence. Phase2 can remain an internal ablation or development diagnostic.

## Task Families

- T1 attached accessory
- T2 container insertion
- T3 surface decal
- T4 local recolor
- T5 same-color material

The benchmark contains 135 completed tasks. The task manifest is:

```text
data/flowedit_compatible_135/manifest.json
```

## Compared Methods

Main comparison rows:

- Ours-SD3
- SAM-Flow-SD3
- SAM-Flow-FLUX
- FlowEdit-SD3
- FlowEdit-FLUX
- SplitFlow-SD3
- OT-RF enhanced-SD3
- DRFS-SD3
- FireFlow
- RF-Solver-Edit
- ReFLEx
- InstructPix2Pix
- LEDITS++

All methods have 135 completed outputs in:

```text
outputs/flowedit135_metric_runs/
```

The unified metric wrapper links completed outputs into a common directory
layout and records source image, source prompt, target prompt, task family, and
original result path.

## Fairness Rules

1. Use the same 135 task manifest for all methods.
2. Use one completed seed-10 output per method and task.
3. Use the same source image, source prompt, target prompt, and local target
   prompt for all metric rows.
4. Use a fixed evaluation region per task for all methods.
5. Report edit strength and preservation separately rather than collapsing them
   into a single scalar score.

## Fixed Evaluation Region Protocol

Fixed regions are stored in:

```text
data/flowedit_compatible_135/eval_masks/
```

They are named:

```text
{task}_eval_mask.png
```

Generation rule:

- Detect an editable host/object with GroundingDINO phrase prompts.
- Convert the detection box into a task-family-specific soft box region.
- Use manual text-panel boxes only for the four `this_must_be_the_place` tasks.

Audit status:

```text
135 fixed evaluation regions
131 detected regions
4 manual text-panel boxes
0 missing regions
```

Important wording: call these fixed evaluation regions, not exact segmentation
masks.

## Metrics

Main edit metrics:

- `local_clip_t`: CLIP score between the local crop and a local target phrase.
- `clip_direction_similarity`: CLIP image-text direction agreement.
- `clip_target_minus_source`: target prompt score minus source prompt score.

Main preservation metrics:

- `bg_lpips`: LPIPS on a background composite.
- `bg_l1`: pixel L1 on the background region.
- `bg_dino_source`: DINO similarity for source vs background composite.
- `bg_ssim_luma`: luma SSIM averaged over the background region.

Background region:

```text
BG = complement(dilate(fixed_eval_region, 31 px))
```

This avoids penalizing natural boundary changes immediately around the edit.

## Tables

Main table:

```text
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_method.csv
```

Supplemental family table:

```text
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
```

Full row table:

```text
experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv
```

## Figures

Main figure:

```text
paper/assets/tradeoff_overall_edit_preservation_compact.png
```

Supplemental:

```text
paper/assets/tradeoff_by_family_edit_preservation.png
experiments/flowedit135_fixedmask_metrics_20260621/review_selected_methods.jpg
experiments/flowedit135_fixedmask_metrics_20260621/review_hard_ours_vs_sam.jpg
data/flowedit_compatible_135/eval_masks/_contact_sheet.jpg
```

## Interpretation Policy

Write:

```text
Ours achieves the strongest non-edit-region preservation and lies on the
edit-preservation Pareto frontier.
```

Do not write:

```text
Ours is the strongest editor on every metric.
Ours is universal image-editing SOTA.
Ours solves text replacement robustly.
```

## Remaining Paper Work

1. Crop or rebuild the qualitative comparison figure for the main paper.
2. Convert the active Markdown notes into the final manuscript format.
3. Decide whether to include the fixed evaluation region contact sheet in the
   appendix.
4. Add family-level breakdown in the supplement.
