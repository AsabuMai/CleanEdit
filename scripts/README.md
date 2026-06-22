# Scripts

Current Phase2 paper-entry scripts:

- `build_phase2_paper_tables.py`
- `build_blind_internal_audit_phase2.py`
- `build_phase2_main_figure_grid.py`
- `fill_phase2_blind_audit_proxy.py`
- `summarize_blind_internal_audit.py`
- `build_efficiency_context_table.py`

The current scope is:

```text
Phase2 = T1-T5, three source cases per family, seeds 10/11/12.
```

Old Core-5/five-case runners, old `red_chair_blue` defaults, and stale visual
audit helpers were moved to:

```text
scripts/obsolete_pre_phase2_lock_2026-06-11/
```

Do not run install/model/GPU/Torch/diffusers work on the master node. Use
Slurm on `a100-01` for heavy operations.

`fill_phase2_blind_audit_proxy.py` fills the audit sheets with metric-guided
proxy ratings for internal plumbing checks. These scores are not human ratings.

`build_phase2_main_figure_grid.py` builds the current compact qualitative grid
from existing Phase2 outputs and does not run model inference.
