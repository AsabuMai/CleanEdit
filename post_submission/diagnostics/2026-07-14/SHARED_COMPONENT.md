# Shared component-wise adaptive controller (2026-07-14)

Purpose: address uneven edit formation across repeated instances and large
subjects without copying content from a donor instance, adding a model forward,
or using backend-specific postprocessing.

For each connected component of the edit support, the shared controller uses
the clean estimates already computed in the RF step:

`gap_c = RMS_c(x0_target - x0_source_step) / RMS_c(x0_target - x_source)`

`weight_c = clamp(1 + gain * max(0, gap_c - dead_zone), min, max)`

The existing preserve-drift budget can disable every component boost. The same
implementation in `dece_core.py` is authoritative for FLUX and can now be made
authoritative for the supported SD3 path with `--shared-core-authoritative`.
Unsupported SD3 combinations fail closed instead of silently falling back.

Fairness properties:

- same formula and flags for SD3 and FLUX
- no extra model forward
- no donor/receiver copy
- no renderer, OCR correction, manual edit, or CPU image repair
- default disabled, preserving previous experiments
- new runs are diagnostic and not paper eligible

Regression job `811767` on `a100-01` completed in 2:44 with exit code 0:

- two component-control unit tests passed
- saved pre-change default PNG, stats, and masks remained bitwise identical
- SD3 legacy and shared-shadow relative error was exactly zero for all steps
- SD3 authoritative output was bitwise identical to legacy
- component control activated and decayed its weight as the target gap closed

FLUX job `811780` contains five paired global/component key cases (`fe_195`,
three hard bucket-6 cases, and `fe_046`) at the same seed and model budget. It is
pending because `h100-01` is currently `down* (Not responding)`; it must not be
moved to A100.
