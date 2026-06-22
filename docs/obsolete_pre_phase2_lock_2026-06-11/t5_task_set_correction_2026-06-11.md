# T5 Task Set Correction 2026-06-11

## Decision

Formal Phase2 T5 is the T5-1-style full-pillow same-color material replacement
set:

- `pillow_same_color_cable_knit`
- `pillow_same_color_cable_knit_grey`
- `pillow_same_color_cable_knit_armchair`

The previously completed `pillow_same_color_linen_panel` and
`pillow_same_color_terry_panel` rows are old center-panel material probes. They
are diagnostic only and must not be used for final Table 3 breadth validation
or any Table 2a/Table 2b summary derived from the corrected Phase2 subset.

## Current Coverage

- Corrected T5 outputs are complete for all three tasks, seeds 10/11/12.
- Internal metrics are complete: 45 rows = 3 tasks x 5 methods x 3 seeds in
  `experiments/support_v3_2026-06-02/table2_t5_internal_metrics.csv`.
- Baseline metrics are complete: 54 rows = 3 tasks x 6 baselines x 3 seeds in
  `experiments/support_v3_2026-06-02/table2_t5_baseline_metrics.csv`.
- Baseline manifest rows are all `complete`, and
  `experiments/support_v3_2026-06-02/e2_t5_baseline_matrix_missing.csv` has no
  missing rows.

## Rebuilt Files

These files now describe the corrected formal T5 task set and should be used
for final Table 3 T5 breadth rows and any scoped Table 2 baseline summaries,
not the old linen/terry panel rows:

- `experiments/support_v3_2026-06-02/table2_t5_internal_metrics.csv`
- `experiments/support_v3_2026-06-02/table2_t5_baseline_metrics.csv`
- `experiments/support_v3_2026-06-02/e2_t5_formal_baseline_manifest.csv`
- `experiments/support_v3_2026-06-02/e2_t5_baseline_matrix_manifest.csv`
- `experiments/support_v3_2026-06-02/normalized_512/t5_eval_assets_manifest.csv`

## Corrected Runner

`scripts/run_phase2_t5_full_gpu01.sh` uses the formal full-pillow T5 task set
above. Slurm job 717149 filled the missing grey/armchair internal and baseline
outputs on `a100-01`, but failed at the metrics stage because CLIP prompt
tokenization exceeded 77 tokens. `scripts/evaluate_paper_metrics.py` now
truncates CLIP text inputs consistently, and metrics-only Slurm job 717295
completed the corrected CLIP/DINO metrics.
