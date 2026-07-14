# Results

This directory contains selected numerical references, not generated images or
model artifacts.

- `submission/`: main no-final-postprocess summaries, the fixed-mask method
  aggregate, and per-sample main metrics.
- `ablations/`: one method-level reference per component, masked-gap,
  multiseed, Pareto, and T4 operator experiment.

The evaluation scripts regenerate the omitted per-family, per-seed, and audit
files. Keeping those derived views out of Git avoids storing the same
measurements in several layouts.

The main comparison must use `ABLATE=no_final_postprocess`. Diagnostic final compositing modes are not part of the fair reported protocol.
