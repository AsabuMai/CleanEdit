# Nine-bucket FLUX evidence rerun (2026-07-14)

## Purpose

This rerun repairs provenance only. It does not change denoising, controller, support, mask, or model-forward logic. The first question is whether the July 4–13 visual conclusions still hold under one auditable FLUX protocol.

Earlier `outputs/evidence_20260714/` runs covered only buckets 1/3/4/6/9 and recorded a dirty worktree while still marking the outputs eligible. They remain historical diagnostics, not the authoritative evidence set.

## Locked protocol

- host: `h100-01.gpu01.cis.k.hosei.ac.jp`
- backend: FLUX.1-dev
- original FlowEdit source images, locked by SHA-256
- support/override masks locked by SHA-256
- seed: 10
- inference steps: 12
- inversion budget (`n_max`): 10
- comparison protocol: single pass
- `ABLATE=no_final_postprocess`
- final mask blend: 0
- final recolor blend: 0
- final texture/overlay post-processing: disabled
- clean Git worktree required for eligibility

The lock file is `data/flowedit_compatible_135/evidence_20260714/nine_bucket_flux_evidence.lock.json`. It contains 41 bucket-case runs. Repeated cases in buckets 2/3 and 6/8 are intentional because the recipes and claims differ.

## Bucket recipes

| Bucket | Runs | Recipe | July conclusion being checked |
|---|---:|---|---|
| 1 INST | 10 | `instance_final_v3` | plural support reaches all instances; fe_195 FLUX quality remains limited |
| 2 SD3 text control | 3 | `flux_canonical_text_control` | FLUX control does not resolve the SD3-specific text limitation |
| 3 FLUX ghost | 3 | `best_single_pass_text_nofinal` | exact-mask/negative/spelling single pass still leaves old structure or wrong glyphs |
| 4 FLUX no-edit | 6 | `kind_budget_final` | simple insertions improve; backlit whole-object recolors remain weak |
| 5 SD3 leak control | 4 | `flux_canonical_recolor_control` | FLUX remains the unchanged control for the SD3 halo mechanism |
| 6 FLUX background replacement | 7 | `zero_expand_outside_lock_final` | several backgrounds recover; meditation whole-subject transforms remain limited |
| 7 half conversion | 2 | `full_body_mask_zero_expand` | full-body support removes the head-only mask failure |
| 8 melt/scale | 5 | `flux_support_control_zero_expand` | FLUX remains the recolor stability control; zero material expansion controls scale |
| 9 replacement residue | 1 | `removed_token_support` | FLUX residue remains linked to its write-without-erase limitation |

Bucket 2 is a FLUX control only. It cannot by itself prove the SD3 limitation; that conclusion remains tied to the separate fixed-input SD3 evidence.
Bucket 8 likewise does not inject SD3-only `--edit-color-mask-image`; FLUX uses its own locked support-control path.

## Eligibility metadata

Every `metadata.json` must contain and pass:

- effective configuration and its SHA-256;
- source image path, size, and SHA-256;
- Git commit and tracked/untracked dirty state;
- evidence bucket, recipe, and lock SHA-256;
- configured final-postprocess fields and `postprocess_applied`;
- `evaluation_eligible`, `eligibility_reasons`, and comparison protocol.

Dirty worktrees are now ineligible. The verifier recomputes configuration, manifest, source-image, and support-asset hashes instead of only checking that fields exist.
