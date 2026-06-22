# RF h-Edit Project Map

This map reflects the Phase2 lock on 2026-06-11.

## Current Scope

```text
Phase2 = T1-T5, three source cases per family, seeds 10/11/12.
```

The current task map is recorded in:

```text
PHASE2_LOCK_2026-06-11.md
experiments/support_v3_2026-06-02/phase2_t1_t5_task_map_2026-06-11.csv
```

`red_chair_blue` and the old five-canonical-case Core-5 tables are obsolete for
the current paper evidence.

## Active Code

- `run_edit_sd3.py`: main SD3 RF editing CLI.
- `sd3_hrec.py`: RF editing orchestration and ODE controller.
- `sd3_model_ops.py`: SD3 velocity calls, Q/K/V injection, feature extraction,
  x0 prediction, and source inversion.
- `edit_preprocess.py`: image preprocessing, normalized box helpers, and mask
  builders used by the CLI.
- `spatial_masks.py`: mask morphology, support fusion, mask statistics, and
  external mask/image loading.
- `operation_support_v3.py`: operation-aware support proposal.
- `guidance_fields.py`: decode-space color/reference losses and guidance-field
  smoothing/projection/RMS limiting.
- `recolor_projection.py`: recolor clean-projection and debug helpers.

## Active Paper/Audit Scripts

- `scripts/build_phase2_paper_tables.py`
- `scripts/build_blind_internal_audit_phase2.py`
- `scripts/summarize_blind_internal_audit.py`
- `scripts/build_efficiency_context_table.py`

Historical runners may still exist for traceability, but the files above are
the current Phase2 paper-entry scripts.

## Active Experiments

Root:

```text
experiments/support_v3_2026-06-02/
```

Current final files:

- `phase2_paper_tables_2026-06-11.md`
- `phase2_tables_audit_2026-06-11.json`
- `table1_phase2_t1_t5_main_final.csv`
- `table2a_phase2_sd3_common_subset_final.csv`
- `table2b_phase2_native_context_final.csv`
- `phase2_t1_t5_family_breakdown_final.csv`
- `efficiency_context_2026-06-11.md`
- `blind_internal_audit_phase2_t1_t5_2026-06-11/`

## Archive Locations

- `obsolete_pre_phase2_lock_2026-06-11/`: old root README/map.
- `experiments/support_v3_2026-06-02/obsolete_pre_phase2_lock_2026-06-11/`: old Core-5/five-case outputs and blind-audit package.
- `paper/obsolete_pre_phase2_lock_2026-06-11/`: old paper drafts/design docs.
- `docs/obsolete_pre_phase2_lock_2026-06-11/`: old notes/todo/design docs.
- `scripts/obsolete_pre_phase2_lock_2026-06-11/`: old table/blind-audit builders and pycache.

## Cleanup Rule

Use the Phase2 task map and active final files above for current reporting.
Everything that still describes Table 1 as five Core-5 rows, uses
`red_chair_blue` as T4, or treats Phase2 as a separate Table 3 breadth stage is
obsolete unless explicitly being cited as history.
