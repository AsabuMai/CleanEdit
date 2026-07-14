# Nine-bucket FLUX evidence audit (2026-07-14)

This audit re-runs the locked post-submission FLUX cases before any controller
change. It is evidence for the nine-bucket engineering campaign, not a paper
result.

## Locked protocol

- host: `h100-01.gpu01.cis.k.hosei.ac.jp`
- code commit: `db47f44812f5a257075da18142022e7340a01192`
- lock SHA-256: `e0c25e2cd301a77919ac15d1ddf3a8032c3debbb339c2d859b6fb6eba68a0be3`
- source inputs: original files, each bound by size and SHA-256
- budget: 12 inference steps, `n_max=10`, seed 10
- final mode: `no_final_postprocess`
- one model pass; no renderer, OCR correction, manual edit, or CPU repair
- expected results: 41 across all nine evidence buckets
- output root: `outputs/evidence_20260714_clean_v4/`

The lock records the manifest entry hash, source-image hash, support-asset
hashes, bucket, recipe, complete batch recipe, and both configuration hashes.
Each result records the clean Git state, CLI run-config hash, batch-recipe hash,
lock hash, eligibility, and actual postprocess state. The aggregate verifier
recomputes these constraints and fails closed.

## Execution record

| Job | Result | Reason |
| --- | --- | --- |
| 812217 | rejected | Bucket 8 passed an SD3-only argument to the FLUX entrypoint. |
| 812256 | rejected | The runner resolved the old project root, so results came from a dirty worktree. |
| 812323 | rejected | The runner's explicit cache directory pointed at the clean worktree rather than the shared H100 cache. |
| 812328 | rejected after visual audit | 41/41 provenance checks passed, but the lock did not bind batch-only recipe settings; bucket 6 omitted the r5 duplicate-figure negative prompt and `fe_157` regressed to a black silhouette. |
| 812713 | accepted | Clean-root rerun with the batch recipe itself hashed and checked against the lock. |

Rejected attempts are retained as audit history and are not counted as
evidence.

## Machine verification

Job 812713 completed with exit code `0:0` in 16:29. The aggregate verifier
accepted 41/41 runs with no errors. Every result binds commit `db47f44`, the
lock above, both configuration hashes, original source and support assets, a
clean worktree, seed 10, 12 steps, `n_max=10`, single-pass comparison, and
`postprocess_applied=false` / `final_postprocess_mode=none`.
The resulting `verification_report.json` has SHA-256
`21d16e01b53b842dae65faa99d1a074559161bb0bda0dce7f4a28341084ffac7`.

As a reproduction check, all 34 non-bucket-6 images are bitwise identical to
the rejected v3 images; only the intentionally corrected bucket-6 recipe
changes pixels. Twenty-two authoritative results in buckets 1/4/6/9 are also
bitwise identical to their corresponding historical no-final results: all ten
bucket-1 cases, all six bucket-4 cases, the five original r5 bucket-6 cases,
and bucket 9. Bucket 3 uses the stronger locked single-pass diagnostic recipe,
so it is judged visually rather than claimed as a bitwise reproduction.

## Visual conclusion audit

| Bucket | Cases | Locked prior conclusion | H100 no-final finding | Verdict |
| --- | ---: | --- | --- | --- |
| 1, plural instances | 10 | Support reaches all instances; `fe_195` crown quality remains limited. | Paired cats/dogs, parrots, and penguins are all reached. `fe_195` gives one large graphic crown and one tiny weak crown; `fe_036` retains most blueberry-like fruit; `fe_197` loses fine structure. | Instance-selection mechanism confirmed; not a clean 10/10 visual closure. |
| 2, SD3 text control | 3 | FLUX is only a control for the SD3 glyph-formation limitation. | FLUX forms readable `ECCV` and `CVPR`, but the `ICCV` sign is malformed. | Useful FLUX capability control only; it neither fixes nor independently proves the SD3 limitation. |
| 3, FLUX text ghost | 3 | Old glyph structure or incorrect spelling persists in a single pass. | `fe_119` produces malformed letters, `fe_134` corrupts the note without forming `ECCV`, and `fe_238` retains `STOP`. | Confirmed unresolved backbone/trajectory limitation. |
| 4, FLUX under-edit | 6 | Simple insertions improve; backlit whole-object transforms stay weak. | Cocktail and whipped cream fire clearly. Owl, horse, and pizza change only partially; the silhouetted sailboat stays effectively unchanged. | Prior partial-fix conclusion confirmed; whole-object under-edit remains open. |
| 5, SD3 halo control | 4 | The repaired halo mechanism is SD3-specific; FLUX is a control. | Both bicycles, the cupcake, and iguana recolor locally without a broad background halo. | FLUX control is clean; this run is not evidence that the separate SD3 repair is complete. |
| 6, FLUX background repaint | 7 | Zero expansion plus outside lock protects several backgrounds; hard whole-subject transforms remain limited. | Background and scale are substantially protected. `fe_157` becomes a coherent statue, but castle and bird conversions remain partial; both meditation cases are half-source/half-target; the lion/bear cases edit mainly the head. | Background mechanism confirmed; broad visual closure rejected. This is the main shared-controller/mask problem. |
| 7, half-subject transform | 2 | The fixed full-body support removes the head-only failure. | Both corgi silhouettes are edited from head through body, although source texture and local artifacts remain. | Full-body support fix confirmed; output quality remains conditional. |
| 8, melt/scale control | 5 | FLUX is the recolor control; zero expansion stabilizes material-transform scale. | Shapes and scale stay stable, but recolor/material formation is weak: `fe_128` changes only a wing patch, butterflies remain partly source-colored, and the bird is only partly gold. | Stability claim confirmed; complete transformation claim rejected. |
| 9, replacement residue | 1 | SD3 was repaired; FLUX residue remains tied to the old-content limitation. | The top hat is added, but crown prongs and metal structure remain visibly underneath it. | FLUX residue remains unresolved; the SD3 claim still relies on separate SD3 evidence. |

## Defensible baseline before new work

The fair no-final baseline supports mechanism-level claims for plural-instance
selection, full-body support, simple insertion budgets, and outside-background
locking. It does not support saying that all nine buckets are visually closed.
The remaining controller work is concentrated in per-component under-edit,
whole-subject source attachment, operation-level mask geometry, and removed
source content. Exact SD3 small text and FLUX old-text erasure remain declared
limitations; renderer, OCR correction, and multi-backbone pipelines stay out of
the single-pass method track.

No algorithm or controller value was changed for this audit. New controller
experiments begin only after this baseline is accepted.
