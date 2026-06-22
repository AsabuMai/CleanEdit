# Phase2 Mask Sensitivity Decision 2026-06-11

## Decision

Do not run a Phase2 mask sensitivity sweep immediately.

## Rationale

- The Phase2 tables are already complete and audited:
  `experiments/support_v3_2026-06-02/phase2_tables_audit_2026-06-11.json`.
- The Phase2 blind internal audit package has now been filled with Codex
  metric-guided proxy ratings and summarized:
  `experiments/support_v3_2026-06-02/blind_internal_audit_phase2_t1_t5_2026-06-11/blind_internal_audit_summary.md`.
- The old mask sensitivity sweep was tied to the obsolete five-case Core-5
  framing and has been archived. It should not be cited as current Phase2
  evidence.
- A proper Phase2 sweep should be a fresh artifact with Phase2 task mapping,
  15 cases, seeds 10/11/12, and current table method names. It should not reuse
  the old Core-5 filenames or table language.

## Trigger To Run Later

Run a Phase2 mask sensitivity sweep only if at least one of these is true:

- The final paper needs an explicit reviewer-defense statement that the
  outside-mask preservation ranking is stable under eroded/dilated masks.
- Human visual audit scores conflict with the metric ranking and we need to
  rule out mask-boundary sensitivity.
- A reviewer or advisor specifically asks for mask robustness on the full
  Phase2 15-case set.

## Required Scope If Run

Use:

```text
T1-T5 x 3 cases per family x seeds 10/11/12 x 4 internal methods
```

Methods:

- `base_only`
- `direct_target`
- `adaptive_full_generic_support`
- `support_v3_controller_rmsgap`

Variants:

- eroded mask
- base mask
- dilated mask

Outputs should use Phase2 names, for example:

- `mask_sensitivity_phase2_t1_t5_summary.md`
- `mask_sensitivity_phase2_t1_t5_summary.csv`
- `mask_sensitivity_phase2_t1_t5_audit.json`

## Compute Note

If the sweep only recomputes image/mask metrics from existing outputs, it can be
treated as light aggregation. If it touches CLIP/DINO/Torch/GPU-backed metrics,
run it on `a100-01` through Slurm rather than on the master node.
