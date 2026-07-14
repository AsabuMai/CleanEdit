# Results

This directory contains compact numerical evidence, not generated images or model artifacts.

- `submission/`: main no-final-postprocess and fixed-mask summaries, per-sample main metrics, and audit metadata.
- `ablations/multiseed/`: seeds 10, 11, and 12.
- `ablations/components/`: controller component ablation excluding T4 recolor.
- `ablations/masked_gap/`: masked-gap component analysis.
- `ablations/pareto/`: edit/preservation trade-off sweep.
- `ablations/t4_operator/`: T4 recolor operator and color-specific metrics.

The main comparison must use `ABLATE=no_final_postprocess`. Diagnostic final compositing modes are not part of the fair reported protocol.
