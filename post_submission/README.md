# Post-submission Research

This directory separates work performed after the submitted paper from the
frozen paper-facing evidence documented in `docs/SUBMISSION_FREEZE.md`.

## Nine-bucket campaign

- Period: 2026-07-04 through 2026-07-06.
- Commit range: `c0fd299^..6b80254` (10 commits).
- Detailed log: `docs/fix_log_20260704.md`.
- Visual audit: `docs/visual_audit_fe135_20260704.md`.
- Slurm launchers: `post_submission/nine_bucket/slurm/`.

Closed mechanisms include plural-instance support, replacement residue,
recolor leakage, mask-expansion background repaint, and half-subject material
transfer. Exact SD3 small text, FLUX exact replacement ghosting, and difficult
whole-subject material transforms remain limitations.

## Fairness rule

Main-comparison outputs must use the same task inputs, inference budget, and
evaluation. A method-specific renderer, OCR module, manual edit, second model
pass, or CPU postprocess is diagnostic only. Such metadata must set
`evaluation_eligible=false` and `paper_use=false`.

OCR may score a frozen output, but it may not alter a prompt, mask, latent, or
image in the main comparison. Multi-pass and multi-backbone hybrids are tracked
as separate diagnostics. They may enter a comparison only when every compared
method receives the same number of model passes and the same forward budget;
they are never merged into the single-pass CleanEdit row.

The shared operation mask policy is also fixed before looking at a case:
additions and shape changes receive one finite latent-space expansion;
recolor, material, and texture-only edits receive zero expansion; replacements
union compact removed-token support. The rule and thresholds are identical for
SD3 and FLUX. Per-image mask-radius or removed-area tuning is not eligible.

## Later diagnostics

The 2026-07-13 shared-controller, fe_195, and fe_186 probes live under
`post_submission/diagnostics/2026-07-13/`. They do not add a closed visual case
to the nine-bucket count.
