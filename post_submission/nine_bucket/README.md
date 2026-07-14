# Nine-bucket Campaign

The nine visual-defect buckets were investigated after paper submission. Their
launchers are archived in `slurm/`; generated results remain in their original
`outputs/` locations so historical metadata and paths stay intact.

`staging_snapshot/` contains the temporary transfer copies used during the
campaign. It is retained for traceability and is not an active code path.

| Bucket | Topic | Final status |
| --- | --- | --- |
| 1 | plural instances | mechanism fixed |
| 2 | SD3 text blank/collapse | localization fixed; glyph formation limitation |
| 3 | FLUX replacement ghost | unresolved limitation |
| 4 | FLUX under-edit | simple cases fixed; hard whole-subject cases limited |
| 5 | SD3 recolor leakage | fixed |
| 6 | background repaint | fixed where mask/outside lock applies; hard transforms limited |
| 7 | half-subject transform | fixed |
| 8 | melt/scale | core cases fixed |
| 9 | replacement residue | SD3 fixed; FLUX residue shares bucket 3 limitation |

The source of truth is `docs/fix_log_20260704.md`.
