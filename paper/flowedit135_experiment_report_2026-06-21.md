# FlowEdit-135 Experiment Report 2026-06-21

## 1. Purpose

The main experiment validates a preservation-first localized editing claim:

```text
Our method achieves the strongest non-edit-region preservation on FlowEdit-135
while remaining on the edit-preservation Pareto frontier.
```

This is not a claim of maximum edit amplitude. The evaluation intentionally
separates edit strength from non-edit-region preservation.

## 2. Scope

Dataset:

```text
data/flowedit_compatible_135/manifest.json
```

Completed outputs:

```text
outputs/flowedit135_metric_runs/
```

Metric output:

```text
experiments/flowedit135_fixedmask_metrics_20260621/
```

Status:

- 135 tasks.
- 13 methods.
- 1755 complete metric rows.
- 0 missing outputs.
- 135/135 fixed evaluation regions.
- 135/135 local target prompts.

## 3. Methods

Compared methods:

| Method | Notes |
| --- | --- |
| Ours-SD3 | proposed preservation-first method |
| SAM-Flow-SD3 | strongest local CLIP edit score |
| SAM-Flow-FLUX | strong preservation among FLUX baselines |
| FlowEdit-SD3 | SD3 FlowEdit baseline |
| FlowEdit-FLUX | FLUX FlowEdit baseline |
| SplitFlow-SD3 | SD3 SplitFlow baseline |
| OT-RF enhanced-SD3 | enhanced OT-RF variant |
| DRFS-SD3 | Delta Rectified Flow Sampling |
| FireFlow | FLUX baseline |
| RF-Solver-Edit | FLUX baseline |
| ReFLEx | FLUX baseline |
| InstructPix2Pix | traditional diffusion editing baseline |
| LEDITS++ | traditional diffusion editing baseline |

## 4. Metric Protocol

Each task has a fixed evaluation region. The background region is defined as:

```text
BG = complement(dilate(fixed_eval_region, 31 px))
```

Edit metrics:

- `local_clip_t`
- `clip_direction_similarity`
- `clip_target_minus_source`

Preservation metrics:

- `bg_lpips`
- `bg_l1`
- `bg_dino_source`
- `bg_ssim_luma`

## 5. Main Quantitative Result

| Method | local CLIP-T up | CLIP dir up | BG-LPIPS down | BG-L1 down | BG-DINO up | BG-SSIM up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours-SD3 | 0.2447 | 0.2757 | **0.0054** | **0.0029** | **0.9984** | **0.9939** |
| SAM-Flow-SD3 | **0.2543** | **0.3083** | 0.0363 | 0.0236 | 0.9745 | 0.9106 |
| SplitFlow-SD3 | 0.2517 | 0.2864 | 0.0789 | 0.0380 | 0.9614 | 0.8869 |
| FlowEdit-SD3 | 0.2500 | 0.2798 | 0.0793 | 0.0383 | 0.9609 | 0.8862 |
| SAM-Flow-FLUX | 0.2238 | 0.2302 | 0.0179 | 0.0164 | 0.9875 | 0.9479 |
| OT-RF enhanced-SD3 | 0.2295 | 0.2287 | 0.0296 | 0.0206 | 0.9846 | 0.9327 |
| FireFlow | 0.2516 | 0.2402 | 0.1418 | 0.0578 | 0.9330 | 0.7851 |
| RF-Solver-Edit | 0.2451 | 0.2357 | 0.1132 | 0.0400 | 0.9521 | 0.8236 |
| ReFLEx | 0.2511 | 0.2560 | 0.1098 | 0.0409 | 0.9613 | 0.7721 |
| InstructPix2Pix | 0.2367 | 0.1734 | 0.1063 | 0.0729 | 0.9621 | 0.7835 |
| LEDITS++ | 0.2433 | 0.2158 | 0.1241 | 0.0650 | 0.9108 | 0.7815 |
| FlowEdit-FLUX | 0.2430 | 0.2484 | 0.1236 | 0.0579 | 0.9438 | 0.8445 |
| DRFS-SD3 | 0.2208 | 0.1689 | 0.2833 | 0.1212 | 0.9128 | 0.5856 |

## 6. Interpretation

The preservation result is very strong:

- Ours has the lowest BG-LPIPS.
- Ours has the lowest BG-L1.
- Ours has the highest BG-DINO.
- Ours has the highest BG-SSIM.

The edit result is moderate:

- Ours has non-trivial local CLIP and CLIP direction scores.
- SAM-Flow-SD3 is the strongest editor by local CLIP-T and CLIP direction.
- This separation creates the central trade-off story.

## 7. Figure

Main trade-off figure:

```text
paper/assets/tradeoff_overall_edit_preservation_compact.png
```

Figure message:

```text
Ours occupies the high-preservation region and stays on the Pareto frontier,
while more aggressive editors move rightward but downward.
```

## 8. Mask Audit

Fixed evaluation region audit:

```text
data/flowedit_compatible_135/eval_masks/_audit.csv
data/flowedit_compatible_135/eval_masks/_contact_sheet.jpg
```

Summary:

- 135 total fixed evaluation regions.
- 131 detected regions.
- 4 manual text-panel boxes.
- 14 regions with area greater than 0.8.

These masks should be described as fixed evaluation regions, not exact
segmentation masks.

## 9. Limitations To Carry Into The Manuscript

- Text-like T3 surface-decal edits are often under-edited by Ours.
- The method should be framed as preservation-first, not edit-maximum.
- Some fixed evaluation regions are large, so report mask area statistics or
  include the contact sheet in the appendix.
- The benchmark mixes SD3, FLUX, and traditional editing methods; avoid
  overclaiming a strict same-backbone SOTA ranking.

## 10. Manuscript Claim

Recommended sentence:

```text
On FlowEdit-135, our method substantially reduces non-edit-region drift and
occupies a favorable point on the edit-preservation Pareto frontier, rather
than simply maximizing edit strength.
```
