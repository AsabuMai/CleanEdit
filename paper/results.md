# Submitted Results

Frozen source of truth for the submitted paper: FlowEdit-135 no-final-restore main-table refresh,
completed on 2026-06-29, combined with the peer-reviewed baseline evaluation
completed on 2026-06-21.

Use these files for manuscript writing:

```text
experiments/norestore_metrics/metrics.csv
experiments/norestore_tradeoff/summary_main_no_samflow.csv
experiments/norestore_tradeoff/summary_by_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/metrics.csv
experiments/flowedit135_fixedmask_metrics_20260621/summary_by_family_method.csv
experiments/flowedit135_fixedmask_metrics_20260621/metric_audit.json
data/flowedit_compatible_135/eval_masks/_audit.csv
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png
```

## Evidence Status

- Dataset: FlowEdit-compatible 135 tasks.
- Main-table methods: 12, after excluding SAM-Flow as concurrent arXiv context.
- Main-table metric rows: 1620.
- Main-table complete rows: 1620/1620.
- No-final-restore Ours rows: 270/270, completed on `a100-01` by Slurm job
  `769100`.
- Baseline metric job: `745496`, completed on `a100-01`, elapsed `01:52:28`.
- Fixed evaluation region rows: complete.
- Local target prompt rows: complete.

## Main Result

The current evidence supports a preservation-first claim:

```text
Our method achieves the strongest non-edit-region preservation while remaining
on the edit-preservation Pareto frontier.
```

This should not be written as strongest edit amplitude. The results show that
SplitFlow-SD3 and other editing baselines can produce stronger edit-success
scores, while our method changes the background far less. The main Ours rows
are no-final-restore runs and do not rely on final pixel-level source
compositing.

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
| Ours-FLUX | 135 | 0.2480 | 0.0517 | **0.0145** | 0.0042 | **0.9988** | 0.9914 |
| Ours-SD3 | 135 | 0.2408 | 0.0571 | 0.0180 | **0.0028** | 0.9985 | **0.9939** |
| SplitFlow-SD3 | 135 | **0.2517** | **0.0636** | 0.0789 | 0.0380 | 0.9614 | 0.8869 |
| FlowEdit-SD3 | 135 | 0.2500 | 0.0624 | 0.0793 | 0.0383 | 0.9609 | 0.8862 |
| FireFlow | 135 | 0.2516 | 0.0604 | 0.1418 | 0.0578 | 0.9330 | 0.7851 |
| ReFLEx | 135 | 0.2511 | 0.0573 | 0.1098 | 0.0409 | 0.9613 | 0.7721 |
| RF-Solver-Edit | 135 | 0.2451 | 0.0533 | 0.1132 | 0.0400 | 0.9521 | 0.8236 |
| OT-RF enhanced-SD3 | 135 | 0.2295 | 0.0247 | 0.0296 | 0.0206 | 0.9846 | 0.9327 |
| InstructPix2Pix | 135 | 0.2367 | 0.0172 | 0.1063 | 0.0729 | 0.9621 | 0.7835 |
| LEDITS++ | 135 | 0.2433 | 0.0525 | 0.1241 | 0.0650 | 0.9108 | 0.7815 |
| FlowEdit-FLUX | 135 | 0.2430 | 0.0516 | 0.1236 | 0.0579 | 0.9438 | 0.8445 |
| DRFS-SD3 | 135 | 0.2208 | 0.0131 | 0.2833 | 0.1212 | 0.9128 | 0.5856 |

## Interpretation

The table supports three paper-facing points.

First, the two no-final-restore Ours variants occupy the strongest
preservation region. Ours-FLUX is best on BG-LPIPS and BG-DINO, while Ours-SD3
is best on BG-L1 and BG-SSIM.

Second, the method is not a no-op. Its CLIP delta and direction scores are in
the same range as several editing baselines, although it is not the strongest
on local CLIP-T.

Third, the strongest main-table edit score belongs to SplitFlow-SD3, with
FlowEdit-SD3, FireFlow, and ReFLEx close behind. This shows that the evaluation
captures the expected edit-preservation trade-off instead of rewarding only
conservative outputs.

## Family-Level Result

Across all five task families, the Ours variants should be reported as
preservation-first methods. Use the no-final-restore family summary when it is
available; until then, do not reuse the older restored family-rank wording
verbatim.

Edit strength varies by family. Ours is especially conservative on T3 text and
surface-decal tasks, where baselines such as FlowEdit-SD3 and SplitFlow-SD3
can rewrite target text more aggressively but also induce larger non-edit
drift.

## Qualitative Reading

Representative review sheets:

```text
experiments/flowedit135_fixedmask_metrics_20260621/review_selected_methods.jpg
experiments/norestore_tradeoff/tradeoff_overall_edit_preservation_compact.png
```

The submitted qualitative selection uses the no-final-restore outputs. Do not
replace it with post-submission nine-bucket outputs without defining a new
revision. The qualitative message is that Ours keeps backgrounds
tightly preserved and changes the intended region cautiously, while stronger
editing baselines often alter more of the source image.

## Claim Boundary

Supported:

```text
On FlowEdit-135, the proposed method provides the strongest non-edit-region
preservation among the main peer-reviewed baseline set and lies on the
edit-preservation Pareto frontier.
```

Not supported:

```text
The method is the strongest editor on every task.
The method is broad image-editing SOTA.
The method solves text replacement robustly.
SAM-Flow is part of the main peer-reviewed baseline set.
```
