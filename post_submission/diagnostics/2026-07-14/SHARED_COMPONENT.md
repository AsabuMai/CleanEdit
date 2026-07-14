# Shared component-wise adaptive controller (2026-07-14)

Purpose: address uneven edit formation across repeated instances and large
subjects without copying content from a donor instance, adding a model forward,
or using backend-specific postprocessing.

For each connected component of the edit support, the shared controller uses
the clean estimates already computed in the RF step:

`gap_c = RMS_c(x0_target - x0_source_step) / RMS_c(x0_target - x_source)`

`weight_c = clamp(1 + gain * max(0, gap_c - dead_zone), min, max)`

The existing preserve-drift budget can disable every component boost. The same
implementation in `dece_core.py` is authoritative for FLUX. On SD3,
`--adaptive-component-control` automatically enables the supported authoritative
shared-core path; unsupported combinations fail closed instead of silently
falling back.

Fairness properties:

- same formula and flags for SD3 and FLUX
- no extra model forward
- no donor/receiver copy
- no renderer, OCR correction, manual edit, or CPU image repair
- default disabled, preserving previous experiments
- new runs are diagnostic and not paper eligible

Regression job `812835` on `a100-01` completed in 2:19 with exit code 0:

- two component-control unit tests passed
- saved pre-change default PNG, stats, and masks remained bitwise identical
- SD3 legacy and shared-shadow relative error was exactly zero for all steps
- SD3 authoritative output was bitwise identical to legacy
- the component flag activated the SD3 authoritative shared controller

FLUX job `812830` on `h100-01` completed in 4:43 with exit code 0. It ran five
paired global/component key cases (`fe_195`, three hard bucket-6 cases, and
`fe_046`) at seed 10, 12 inference steps, and `n_max=10`, with final
postprocessing disabled. All five component runs found at least one connected
component and applied a non-unit component weight; the maximum observed weight
was 1.55.

This is a mechanism result, not a nine-bucket closure. Visual comparison found
that the component controller did not resolve the LEGO-castle under-edit, the
two whole-person material transforms, the crown-to-hat residue, or the uneven
two-parrot crown quality. It changes the hard cases without copying a donor and
without extra forwards, but these outputs remain diagnostics and the unresolved
visual failures remain limitations.
