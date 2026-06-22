# Limitations

- The method depends on reliable spatial support for local edits.
- Current implementation is SD3-specific.
- The revised strict Phase 1 headline evidence covers five Core-5 rows:
  attached accessory, container-constrained insertion, surface decal, local
  recolor, and same-color material replacement. The suite is still a controlled
  diagnostic set, not a large-scale benchmark. Exposed-object removal is
  reported separately as an E5 boundary probe.
- Container-constrained insertion is covered by `bowl_apple_inside`, but the
  current `inside_container` relation is still a simple geometry prior derived
  from the grounded bowl mask. Do not overstate it as general 3D container
  reasoning or robust free-space placement.
- Same-color surface material editing is covered by
  `pillow_same_color_cable_knit`. It should be described as localized
  same-color material replacement with visual/material gates, not as broad
  material transfer.
- Removal is only partially covered. `backpack_remove_toy_charm` removes the
  intended dangling charm, but the zipper/fabric region occluded by the charm is
  locally smoothed. More difficult occluded-object removal requiring host or
  background completion, such as `dog_remove_tennis_ball`, remains outside the
  main claim.
- Accurate support does not guarantee object erasure or surface completion.
  Sticker, magnet, and letter-removal probes show residual marks, transformed
  objects, or nearby-object damage even when localization is plausible.
- High-confidence completion prior helps only under restricted conditions. The
  `laptop_remove_sticker` probe supports planar-surface completion, while
  cluttered or semantic hosts should be gated off.
- Replacement target formation is not broadly solved. The whiteboard red-star
  probe shows that a strong non-glyph color/shape target can work, but precise
  glyph replacement, small tags, and dog-ball replacement remain unreliable.
- Recolor/appearance editing is represented by `red_chair_blue`, which passed
  the strict Phase 1 visual audit. Describe it as a localized recolor probe,
  not evidence of general appearance editing.
- `support_v3_fixed` is the Fixed DeCE component-control row. Its gap to
  DeCE-RF is modest in the base comparison, so feedback control should be
  presented as tradeoff/stabilization evidence through stress/Pareto curves,
  not as a standalone headline robustness claim.
- Mask sensitivity has been checked under eroded/base/dilated fixed evaluation
  masks and the Core-5 outside-L1 ranking is stable, but outside-mask metrics
  should still be interpreted together with visual audits for boundary-heavy
  edits.
- The efficiency table is a context table. Internal SD3 rows record runtime and
  peak memory, while several external baseline producers did not record those
  fields. Do not claim broad efficiency superiority from incomplete external
  runtime metadata.
- External baselines must be compared only under matched prompts, seeds,
  resolution, backbone assumptions, and mask inputs. Native FLUX rows are
  contextual, and OT-RF/DVRF are currently smoke-verified but strict-matrix
  pending. Older core-4 baseline artifacts should be labeled as contextual
  evidence unless rerun under the current evidence protocol.
