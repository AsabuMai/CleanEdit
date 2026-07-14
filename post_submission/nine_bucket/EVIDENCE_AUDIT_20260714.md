# Nine-bucket FLUX evidence audit (2026-07-14)

This audit re-runs the locked post-submission FLUX cases before any controller
change. It is evidence for the nine-bucket engineering campaign, not a paper
result.

## Locked protocol

- host: `h100-01.gpu01.cis.k.hosei.ac.jp`
- code commit: `TO_FILL`
- lock SHA-256: `TO_FILL`
- source inputs: original files, each bound by size and SHA-256
- budget: 12 inference steps, `n_max=10`, seed 10
- final mode: `no_final_postprocess`
- one model pass; no renderer, OCR correction, manual edit, or CPU repair
- expected results: 41 across all nine evidence buckets
- output root: `outputs/evidence_20260714_clean_v4/`

The lock records the manifest entry hash, source-image hash, support-asset
hashes, bucket, and recipe. Each result records the clean Git state, run-config
hash, lock hash, eligibility, and actual postprocess state. The aggregate
verifier recomputes these constraints and fails closed.

## Execution record

| Job | Result | Reason |
| --- | --- | --- |
| 812217 | rejected | Bucket 8 passed an SD3-only argument to the FLUX entrypoint. |
| 812256 | rejected | The runner resolved the old project root, so results came from a dirty worktree. |
| 812323 | rejected | The runner's explicit cache directory pointed at the clean worktree rather than the shared H100 cache. |
| 812328 | rejected after visual audit | 41/41 provenance checks passed, but the lock did not bind batch-only recipe settings; bucket 6 omitted the r5 duplicate-figure negative prompt and `fe_157` regressed to a black silhouette. |
| `TO_FILL` | `TO_FILL` | Clean-root rerun with the batch recipe itself hashed and checked against the lock. |

Rejected attempts are retained as audit history and are not counted as
evidence.

## Machine verification

`TO_FILL`

## Visual conclusion audit

| Bucket | Cases | Locked prior conclusion | H100 no-final finding | Verdict |
| --- | ---: | --- | --- | --- |
| 1, plural instances | 10 | Support reaches all instances; `fe_195` crown quality remains limited. | `TO_FILL` | `TO_FILL` |
| 2, SD3 text control | 3 | FLUX is only a control for the SD3 glyph-formation limitation. | `TO_FILL` | `TO_FILL` |
| 3, FLUX text ghost | 3 | Old glyph structure or incorrect spelling persists in a single pass. | `TO_FILL` | `TO_FILL` |
| 4, FLUX under-edit | 6 | Simple insertions improve; backlit whole-object transforms stay weak. | `TO_FILL` | `TO_FILL` |
| 5, SD3 halo control | 4 | The repaired halo mechanism is SD3-specific; FLUX is a control. | `TO_FILL` | `TO_FILL` |
| 6, FLUX background repaint | 7 | Zero expansion plus outside lock protects several backgrounds; hard whole-subject transforms remain limited. | `TO_FILL` | `TO_FILL` |
| 7, half-subject transform | 2 | The fixed full-body support removes the head-only failure. | `TO_FILL` | `TO_FILL` |
| 8, melt/scale control | 5 | FLUX is the recolor control; zero expansion stabilizes material-transform scale. | `TO_FILL` | `TO_FILL` |
| 9, replacement residue | 1 | SD3 was repaired; FLUX residue remains tied to the old-content limitation. | `TO_FILL` | `TO_FILL` |

## Defensible baseline before new work

`TO_FILL`

No algorithm or controller value was changed for this audit. New controller
experiments begin only after this baseline is accepted.
