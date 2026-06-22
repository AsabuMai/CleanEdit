# Current Results

Current source of truth: FlowEdit-135 full evaluation, completed on
2026-06-21.

Use these files for manuscript writing:

```text
experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/metric_audit.json
data/flowedit_compatible_135/eval_masks/_audit.csv
paper/assets/tradeoff_overall_edit_preservation_compact.png
```

## Evidence Status

- Dataset: FlowEdit-compatible 135 tasks.
- Compared methods: 13.
- Total metric rows: 1755.
- Complete rows: 1755/1755.
- Fixed evaluation region rows: 1755/1755.
- Local target prompt rows: 1755/1755.
- Slurm metric job: `745496`, completed on `a100-01`, elapsed `01:52:28`.

## Main Result

The current evidence supports a preservation-first claim:

```text
Our method achieves the strongest non-edit-region preservation while remaining
on the edit-preservation Pareto frontier.
```

This should not be written as strongest edit amplitude. The results show that
SAM-Flow-SD3 has stronger edit-success scores, while our method has much lower
background drift.

## Overall Method Summary

Key columns:

- `local_clip_t`: edit strength, higher is better.
- `clip_target_minus_source`: target-vs-source CLIP preference, higher is
  stronger edit signal.
- `bg_lpips`: non-edit-region perceptual drift after dilating the edit region,
  lower is better.
- `bg_l1`: non-edit-region pixel drift after dilation, lower is better.
- `bg_dino_source`: source semantic preservation on the background composite,
  higher is better.
- `bg_ssim_luma`: luma structure preservation in the background, higher is
  better.

| Method | n | local CLIP-T up | CLIP delta up | BG-LPIPS down | BG-L1 down | BG-DINO up | BG-SSIM up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours-SD3 | 135 | 0.2447 | 0.0595 | **0.0054** | **0.0029** | **0.9984** | **0.9939** |
| SAM-Flow-SD3 | 135 | **0.2543** | **0.0742** | 0.0363 | 0.0236 | 0.9745 | 0.9106 |
| SplitFlow-SD3 | 135 | 0.2517 | 0.0636 | 0.0789 | 0.0380 | 0.9614 | 0.8869 |
| FlowEdit-SD3 | 135 | 0.2500 | 0.0624 | 0.0793 | 0.0383 | 0.9609 | 0.8862 |
| SAM-Flow-FLUX | 135 | 0.2238 | 0.0338 | 0.0179 | 0.0164 | 0.9875 | 0.9479 |
| OT-RF enhanced-SD3 | 135 | 0.2295 | 0.0247 | 0.0296 | 0.0206 | 0.9846 | 0.9327 |
| FireFlow | 135 | 0.2516 | 0.0604 | 0.1418 | 0.0578 | 0.9330 | 0.7851 |
| RF-Solver-Edit | 135 | 0.2451 | 0.0533 | 0.1132 | 0.0400 | 0.9521 | 0.8236 |
| ReFLEx | 135 | 0.2511 | 0.0573 | 0.1098 | 0.0409 | 0.9613 | 0.7721 |
| InstructPix2Pix | 135 | 0.2367 | 0.0172 | 0.1063 | 0.0729 | 0.9621 | 0.7835 |
| LEDITS++ | 135 | 0.2433 | 0.0525 | 0.1241 | 0.0650 | 0.9108 | 0.7815 |
| FlowEdit-FLUX | 135 | 0.2430 | 0.0516 | 0.1236 | 0.0579 | 0.9438 | 0.8445 |
| DRFS-SD3 | 135 | 0.2208 | 0.0131 | 0.2833 | 0.1212 | 0.9128 | 0.5856 |

## Interpretation

The table supports three paper-facing points.

First, our method has the best preservation metrics by a large margin. It is
the lowest on BG-LPIPS and BG-L1, and the highest on BG-DINO and BG-SSIM.

Second, the method is not a no-op. Its CLIP delta and direction scores are in
the same range as several editing baselines, although it is not the strongest
on local CLIP-T.

Third, the strongest edit score belongs to SAM-Flow-SD3. This is useful rather
than harmful for the paper: it shows that the evaluation captures the expected
edit-preservation trade-off instead of rewarding only conservative outputs.

## Family-Level Result

Across all five task families, Ours-SD3 ranks first on the main preservation
metrics:

- T1 attached accessory: rank 1 on BG-LPIPS, BG-L1, BG-DINO.
- T2 container insertion: rank 1 on BG-LPIPS, BG-L1, BG-DINO.
- T3 surface decal: rank 1 on BG-LPIPS, BG-L1, BG-DINO.
- T4 local recolor: rank 1 on BG-LPIPS, BG-L1, BG-DINO.
- T5 same-color material: rank 1 on BG-LPIPS, BG-L1, BG-DINO.

Edit strength varies by family. Ours is especially conservative on T3 text and
surface-decal tasks, where baselines such as SAM-Flow-SD3 more often rewrite
the target text but also induce larger non-edit drift.

## Qualitative Reading

Representative review sheets:

```text
experiments/flowedit135_fixedmask_metrics_20260621/review_selected_methods.jpg
experiments/flowedit135_fixedmask_metrics_20260621/review_hard_ours_vs_sam.jpg
```

Visual inspection agrees with the metrics. Ours keeps backgrounds nearly fixed
and changes the intended region cautiously. SAM-Flow-SD3, FlowEdit-SD3, and
SplitFlow-SD3 often produce stronger semantic edits, especially for text-like
surface decal tasks, but they modify more of the source image.

## Claim Boundary

Supported:

```text
On FlowEdit-135, the proposed method provides the strongest non-edit-region
preservation and lies on the edit-preservation Pareto frontier.
```

Not supported:

```text
The method is the strongest editor on every task.
The method is broad image-editing SOTA.
The method solves text replacement robustly.
```
