# Current Limitations

- The central claim is preservation-first. The results support superior
  non-edit-region preservation with non-trivial edit signal; they do not prove
  strongest edit amplitude or universal image-editing superiority.
- Ours is conservative on several text-like T3 surface-decal tasks. In cases
  such as gas-station signs and stop-arrow text replacement, FlowEdit-SD3,
  SplitFlow-SD3, FireFlow, and related editing baselines can produce clearer
  target text while changing more of the image. This should be discussed as an
  edit-strength limitation, not hidden.
- The fixed masks are fixed evaluation regions, not exact segmentation masks.
  They are generated from GroundingDINO detections plus task-family box rules,
  with 4 manual text-panel boxes. They are appropriate for fair metric
  evaluation, but the paper should not claim pixel-perfect mask quality.
- Some evaluation regions are large. The audit shows 14 regions with area
  greater than 0.8, mainly groceries, signs, meditation, castle, and pizza
  cases. For these tasks, the background complement is small, so preservation
  metrics should be interpreted together with the mask-area statistics.
- FlowEdit-135 contains heterogeneous task types. T1-T5 are useful for a broad
  localized editing benchmark, but text replacement, material conversion, and
  object insertion place different demands on a method. Family-level results
  should be reported in the supplement.
- The comparison mixes backbone contexts. SD3-based methods, FLUX-based
  methods, and traditional diffusion editing methods are included because they
  are relevant baselines, but the manuscript should avoid overstating a single
  same-backbone SOTA ranking across all rows.
- SAM-Flow is excluded from the main table because it is a recent concurrent
  arXiv preprint with a closely related source-anchored masked-flow design, not
  a peer-reviewed benchmark baseline. If mentioned, treat it as concurrent
  context or appendix material rather than central evidence.
- The main Ours rows use the no-final-restore setting. They do not use final
  pixel-level source compositing, but they remain source-anchored through the
  method's clean-estimate, source-trajectory, and operation-aware preservation
  terms.
- Local CLIP-T and CLIP direction are useful automatic edit-strength signals,
  but they do not fully capture text correctness, relation correctness, or
  human preference. Qualitative figures and hard-case analysis are necessary.
- The present benchmark uses one seed per baseline run. This is sufficient for
  the current completed comparison, but multi-seed variance would strengthen a
  future camera-ready or extended version.
- The older Phase2 proxy blind-audit sheets are internal prechecks, not human
  ratings. Do not use them as human-study evidence unless replaced by actual
  human annotations.
