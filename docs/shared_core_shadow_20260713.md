# SD3 Shared-Core Shadow Validation (2026-07-13)

## Safety contract

`--shared-core-shadow` is opt-in and defaults to off. In shadow mode, the
legacy SD3 controller remains authoritative for `v_total`, `next_z_t`, and the
saved image. The shared core only computes diagnostics from already available
tensors. Unsupported auxiliary branches are reported and skipped.

Default-off runs do not add shadow keys to stats or metadata.

## Compatibility fields

The shared core gained optional compatibility inputs, all defaulting to the
previous FLUX behavior:

- reconstruction feature maps;
- SD3's second preserve-mask application (`base_rec_post_gate`);
- SD3's core/ring edit weighting (`base_edit_post_gate`);
- SD3-style channel-normalized masked RMS.

## A100 validation

All runs used `a100-01`, seed 10, and a fresh output directory.

| Job | Configuration | Mean `v_total` relative error | Max error |
|---|---|---:|---:|
| 808819 | first adaptive shadow | 1.96e-2 | 5.11e-2 |
| 808826 | adaptive off, before rec post-gate | 1.52e-2 | 5.38e-2 |
| 808831 | adaptive off, compatibility complete | 0 | 0 |
| 808834 | adaptive on, compatibility complete | 1.34e-9 | 1.66e-8 |

The edit branch matched at floating-point noise from the first run. The
remaining discrepancy was the legacy SD3 controller's second preserve-gate
application on reconstruction velocity.

Job 808840 reran the same case with shadow disabled. Its `result.png` SHA256
matched the adaptive shadow result exactly:

```text
31991dd227ddab018158b0c2fc9ec89e97e0e8c0a45e964e60d651e903471071
```

## Decision

Keep shadow mode off by default. The common rec/edit/adaptive subset is ready
for broader shadow coverage, but switching SD3's authoritative update to the
shared core should wait until auxiliary fields (text, DDS, color, reference,
completion, and edit-bound control) are represented or explicitly kept in the
SD3 adapter.
