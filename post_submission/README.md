# Post-submission Research

This directory separates work performed after the submitted paper from the
minimal reproduction release. The complete historical lab tree is preserved by
the `archive/full-lab-2026-07-14` Git tag.

## Nine-bucket campaign

- Period: 2026-07-04 through 2026-07-06.
- Commit range: `c0fd299^..6b80254` (10 commits).
- Slurm launchers: `post_submission/nine_bucket/slurm/`.

The campaign introduced plural-instance support, replacement support, recolor
containment, operation-aware expansion, and full-body support. The locked H100
audit in `post_submission/nine_bucket/EVIDENCE_AUDIT_20260714.md` confirms the
mechanisms but rejects a blanket visual-closure claim. Exact SD3 small text,
FLUX exact replacement ghosting, and difficult whole-subject material
transforms remain limitations.

## Fairness rule

Main-comparison outputs must use the same task inputs, inference budget, and
evaluation. A method-specific renderer, OCR module, manual edit, second model
pass, or CPU postprocess is diagnostic only. Such metadata must set
`evaluation_eligible=false` and `paper_use=false`.

OCR may score a frozen output, but it may not alter a prompt, mask, latent, or
image in the main comparison. Multi-pass and multi-backbone hybrids are tracked
as separate diagnostics. They may enter a comparison only when every compared
method receives the same number of model passes and the same forward budget;
they are never merged into the single-pass CleanEdit row.

The shared operation mask policy is also fixed before looking at a case:
additions and shape changes receive one finite latent-space expansion;
recolor, material, and texture-only edits receive zero expansion; replacements
union compact removed-token support. The rule and thresholds are identical for
SD3 and FLUX. Per-image mask-radius or removed-area tuning is not eligible.

## Later diagnostics

The 2026-07-13 shared-controller, fe_195, and fe_186 probes remain in the full
lab archive tag. They do not add a closed visual case to the nine-bucket count.

## 2026-07-15 validation outcome

- Shared controller: A100 job `812835` verified the authoritative SD3 path and
  legacy bitwise parity. H100 job `812830` verified five equal-budget FLUX
  pairs. Component weights activated, but the hard plural, material, and
  replacement examples were not visually resolved.
- Source attachment: the H100 release and control runs produced 18 cases each.
  A100 verifier `813230` proved equal source, prompt, support statistics, and
  masks; 15 active results changed and three inactive results were bitwise
  identical. The fixed 0.35 release did not provide a stable visual benefit and
  worsened the LEGO-castle case, so the mechanism remains default-off.
- Mask policy: A100 job `813097` exercised SD3 replacement and appearance
  paths. The 22 H100 paired outputs were accepted by verifier `813202` at equal
  12-step, `n_max=10`, seed-10, two-attention-forward budgets. Removed-token
  support improved the crown-to-hat case, while zero expansion weakened several
  broad material/plural edits. The uniform geometry rule is validated, but it
  is not evidence that those difficult visual cases are closed.

The H100 model jobs for the source-attachment and mask pairs completed all
requested images but their original Slurm records ended nonzero because the
first inline verifiers referenced obsolete metadata field names. The corrected
verification-only jobs above read the saved outputs and completed with exit
code zero; no model output was regenerated to hide the verifier failures.
