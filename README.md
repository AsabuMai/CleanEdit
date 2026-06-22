# RF h-Edit Project

Current paper method:

```text
DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing
```

Current project lock:

```text
Phase2 = T1-T5, three source cases per family, seeds 10/11/12.
```

Start with `PHASE2_LOCK_2026-06-11.md` for scope and
`CURRENT_PHASE2_STATUS_2026-06-11.md` for what is already complete and what
comes next. The old five-canonical-case Core-5 entry points were archived and
are not current evidence.

## Active Entry Points

- `PHASE2_LOCK_2026-06-11.md`: current scope, task map, table policy, and compute rule.
- `CURRENT_PHASE2_STATUS_2026-06-11.md`: completed artifacts, audit status, and next tasks.
- `PROJECT_MAP.md`: active file map.
- `paper/README.md`: paper-facing current tables/results/figure notes.
- `docs/README.md`: current docs index.
- `docs/todo_2026-06-11.md`: next execution queue.

## Active Evidence

Current artifacts live in:

```text
experiments/support_v3_2026-06-02/
```

Use these as the current paper-facing outputs:

- `phase2_paper_tables_2026-06-11.md`
- `phase2_tables_audit_2026-06-11.json`
- `table1_phase2_t1_t5_main_final.csv`
- `table2a_phase2_sd3_common_subset_final.csv`
- `table2b_phase2_native_context_final.csv`
- `phase2_t1_t5_family_breakdown_final.csv`
- `blind_internal_audit_phase2_t1_t5_2026-06-11/`
- `efficiency_context_2026-06-11.md`

## Active Scripts

- `scripts/build_phase2_paper_tables.py`
- `scripts/build_blind_internal_audit_phase2.py`
- `scripts/summarize_blind_internal_audit.py`
- `scripts/build_efficiency_context_table.py`

The older Core-5/five-case builders and old generated artifacts were moved to
`obsolete_pre_phase2_lock_2026-06-11` folders.

## Compute Boundary

Do not run heavy operations on the master node. Heavy install/model/GPU/Torch/
diffusers work must run on `a100-01` through Slurm:

```bash
srun -p a100 -w a100-01 --gres shard:1 --pty /bin/bash -l
```

The shared environment is:

```text
/cluster/users/grad/2025/25t8103/project/.venv
```

## Current Claim

The conservative claim is that DeCE-RF improves localized edit-preserve behavior
on the locked Phase2 T1-T5 diagnostic set, with preservation and locality
reported alongside edit metrics. E5/removal remains a separate boundary probe,
not part of the Phase2 T1-T5 main tables.
