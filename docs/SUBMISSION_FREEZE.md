# Submission Freeze

The submitted paper evidence is frozen. Post-submission experiments must not be
used to retroactively change the submitted main table, figures, or claims.

## Frozen paper-facing evidence

The FlowEdit-135 main evaluation was completed on 2026-06-29:

- `outputs/norestore_metric_runs/`
- `experiments/norestore_metrics/metrics.csv`
- `experiments/norestore_tradeoff/summary_main_no_samflow.csv`
- `experiments/norestore_tradeoff/summary_by_method.csv`
- `experiments/flowedit135_fixedmask_metrics_20260621/`
- `data/flowedit_compatible_135/eval_masks/`
- `paper/results.md`, `paper/tables.md`, `paper/figures.md`, and
  `paper/limitations.md`

The nearest committed pre-campaign code anchor is `5094602` (2026-06-23). It is
an engineering anchor, not a claim that the exact submitted manuscript state
was fully committed on that date.

## Post-submission boundary

The nine-bucket visual-defect campaign began on 2026-07-04 at `c0fd299` and its
ten local commits end at `6b80254` on 2026-07-06. Everything in that campaign,
plus the 2026-07-13 controller/fairness diagnostics, is post-submission work.

Use `post_submission/README.md` for current research status. Do not mix these
outputs with the frozen paper evidence unless a future revision explicitly
defines a new evaluation and labels it as post-submission.
