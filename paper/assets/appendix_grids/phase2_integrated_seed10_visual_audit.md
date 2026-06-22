# Phase2 Seed10 Integrated Visual Audit

Figure: `phase2_integrated_seed10_audit.png`

Scope: 15 Phase2 tasks, seed 10. Columns are Source, Direct target, Generic support, DeCE-RF, Sam-Flow SD3, and Sam-Flow FLUX.

## Overall Finding

The integrated visual check supports the paper's core claim: DeCE-RF is generally better at localizing the requested edit while preserving the non-edit region. Sam-Flow is a strong and relevant baseline, but on these tasks it often edits the target concept by changing the subject, pose, object geometry, background, or adding extra objects. This is especially visible in T4 recolor and T5 same-color material transfer.

## Task-Level Notes

| Task | DeCE-RF visual check | Sam-Flow SD3 visual check | Sam-Flow FLUX visual check |
|---|---|---|---|
| T1 cat crown | Pass. Crown is localized and cat/background are preserved. | Weak. Crown appears, but cat identity changes strongly. | Mostly pass/weak. Small crown appears with better preservation than SD3, but effect is small and lighting changes. |
| T1 dog bow tie | Pass. Bow tie is localized and dog pose is preserved. | Weak. Bow tie appears, but dog face/fur/pose drift. | Moderate. Bow tie appears; preservation is better than SD3 but still changes dog texture. |
| T1 dog sunglasses | Pass. Sunglasses are placed correctly with good source preservation. | Weak. Sunglasses appear but face geometry/scale drift. | Weak. Sunglasses appear but dog identity/lighting drift. |
| T2 bowl + apple | Pass. Apple is inserted in the bowl with stable table context. | Moderate. Apple appears, with some appearance/shadow drift. | Moderate. Apple appears, but bowl/apple rendering is less natural. |
| T2 white bowl + orange | Pass/weak. Orange appears on tabletop, though placement is not ideal. | Fail/weak. Orange is partly cropped at the bottom and not integrated. | Fail. Orange is tiny and off-target. |
| T2 brown bowl + lemon | Pass. Lemon is in the small bowl and source layout is preserved. | Weak. Lemon appears, but composition introduces extra object changes. | Moderate. Lemon appears, but object shape/lighting drift remains. |
| T3 t-shirt star | Pass. Red star is visible and reasonably integrated with shirt folds. | Moderate. Star appears, but shirt/body changes. | Moderate. Star appears, larger and clearer, but source preservation is weaker. |
| T3 mug heart | Pass. Heart is small, clean, and mug/background are preserved. | Weak. Heart appears, but mug shape/position/background drift. | Weak. Heart appears, but mug geometry and background drift. |
| T3 tote leaf | Weak/moderate. Leaf decal is localized but looks pale/artificial. | Moderate. Leaf target is clearer, but tote/hand/background drift. | Moderate. Leaf target is clearer, but preservation is weaker and texture changes. |
| T4 chair blue | Pass. Chair is recolored blue while background remains stable. | Fail. Chair mostly remains red. | Fail. Blue artifact appears beside chair instead of recoloring chair. |
| T4 mug orange | Pass. Mug is recolored orange and scene stays stable. | Fail/weak. Mug stays largely green or partially edited. | Fail. Little to no intended recolor. |
| T4 vase blue | Pass. Vase becomes blue with source geometry mostly preserved. | Fail. Extra vase/object is introduced. | Fail. Extra blue vase/object is introduced and target is not localized. |
| T5 white pillow knit | Pass/moderate. Cable-knit texture appears while couch is preserved. | Moderate. Knit is clearer but pillow/couch shading changes more. | Moderate. Knit appears but pillow appearance drifts. |
| T5 grey pillow knit | Weak/moderate. Preservation is strong, texture edit is subtle. | Weak. Pillow scale/location and cushion appearance drift. | Weak. Pillow becomes darker/heavier with scene drift. |
| T5 armchair pillow knit | Weak/moderate. Preservation is strong, texture edit is subtle. | Weak. Chair/pillow appearance drift is significant. | Weak. Pillow becomes frilly/changed rather than controlled cable-knit. |

## Paper-Relevant Takeaway

Sam-Flow should remain in E2.3 as the closest source-anchored masked-flow baseline. The seed10 visual audit suggests it is competitive on some insertion/decal cases, but it does not consistently protect non-edit regions. The strongest contrast against DeCE-RF is T4 local recolor and T5 same-color material, where Sam-Flow often either fails the target edit or changes non-edit structure.
