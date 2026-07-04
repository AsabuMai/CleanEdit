# FE-135 Visual Audit — 2026-07-04 (CleanEdit post-submission)

Source: `outputs/paper_current_metric_runs` (canonical paper images), seed 10.
Sheets: `tmp/family_sheets_20260704/` (source | ours_sd3 | ours_flux). T1 audited case-by-case.
Severity: X = severe (must fix / never show), ~ = flawed but passable, blank = OK.

## Defect tags

- INST  = plural/multi-instance target, only one instance edited
- BLANK = SD3 erase-then-blank (text removed, nothing written)
- CCOL  = SD3 content collapse (writes generic/wrong words)
- LDROP = SD3 letter drop/mangle (ICCV->ICV)
- GHOST = FLUX translucent double-render / overlapped glyphs
- NOED  = no visible edit
- LEAK  = color/glow spills outside object (SD3)
- MELT  = structure/face melt, species or identity drift
- BGREP = background replaced/restyled
- HALF  = only part of subject converted (half-transform)
- SCALE = subject shrunk/inflated
- RESID = residue of source object (crown under hat etc.)
- DUP   = duplicated subject hallucinated

## T1 attached_accessory (6) — audited earlier today, worst family per-capita

| case | sd3 | flux |
|---|---|---|
| fe_046 cat crown->top hat | ~ double brim + RESID | X RESID (crown chains fused on hat) |
| fe_094 dog +red top hat | ~ bg smoothed | OK |
| fe_095 dog +jeweled crown | ~ ear MELT spike | OK |
| fe_142 iguana blue+hat | X DUP (2nd blue head in bg) | X NOED recolor + floating hat blob |
| fe_192 parrots both hats | X INST + identity drift | X INST + wrong hat style |
| fe_195 parrots both crowns | X MELT (spiky crest) | ~ INST + sticker-like crown |

## T2 container_insertion (13) — mostly passable

| case | sd3 | flux |
|---|---|---|
| fe_010 beer->cocktail | OK | X NOED (still amber) |
| fe_031/033/034 cakes | OK/~ | OK/~ (033 both weak) |
| fe_036 blueberries->raspberries | ~ partial (only top) | ~ partial |
| fe_180 milk +whipped cream | OK | X NOED |
| fe_186 muffins +strawberries | ~ weak | ~ weak |
| fe_198 cake +cherry | OK | ~ cake body restyled |
| fe_201 pizza pineapple+ham | ~ pineapple looks like orange slices | OK |
| fe_205/206 pizza slices | OK | ~ whole-pizza restyle |
| fe_207 +pepperoni | X center toppings blanked | ~ NOED-ish |
| fe_208 +mushrooms | ~ bottom half overhauled | OK |

Pattern: FLUX systematically under-edits insertions; SD3 edits but locally overshoots.

## T3 surface_decal / text (41) — weakest family, ~5-6/41 fully OK

- Gas-station sign 118-123: SD3 BLANK x5 (one small cvpr); FLUX GHOST on cvpr/iccv/eccv/food, OK on LOVE/FREE.
- Grocery list 131-135: SD3 CCOL x4 (always writes "Grocery/Eggs/Milk"); FLUX right words but paper->grid BGREP + GHOST.
- Neon Luna 164-170: SD3 boxy restyle + spurious "RESTAURANT" + LDROP (ICV/ECV); FLUX keeps neon style, good Sol/CVPR-ish, garbage on "hi" (169), wrong word FREDS (170 heart).
- Billboard 222-225: both mangle key token (SD3 CVPP/ecv//yeeq; FLUX CUPE/EOVE/ELOE ghost). fe_224 SD3 "ICCV" correct.
- Stop sign 235-243: SD3 BLANK x6 (one arrow-icon morph); FLUX clean short words (CVPR/LOVE/BEER/FREE/ICCV; ECOV off by one; SOME for HOME) + field-color contamination on 236/237.
- Sticker 244-246: SD3 right-ish letters but destroys decal art (yellow box); FLUX keeps art but wrong letters (SVPR/ECCP), 246 NOED.
- Neon glass 247-250: SD3 2/4 fully correct sentences (247 home, 250 ICCV), 248 missing word, 249 LDROP; FLUX GHOST x4.

Pattern: SD3 = BLANK/CCOL/LDROP (erase wins, content loses); FLUX = right content, wrong carrier (GHOST + BGREP). Complementary, matches paper's per-backbone framing.

## T4 local_recolor (19) — ~5-6 to fix

| case | sd3 | flux |
|---|---|---|
| fe_000/025 bear black, bus pink | OK | OK / ~ front artifacts |
| fe_004 grass bear black | ~ muddy | ~ purple tint |
| fe_015/016 bikes green/yellow | X LEAK halo on wall | OK |
| fe_017 sailboat white sails | X NOED-ish | X NOED (silhouette hard case) |
| fe_019 owl white | ~ under | ~ under |
| fe_027/028 butterflies | X MELT | OK / ~ |
| fe_029 flower red | OK | OK |
| fe_067 kitten black | ~ under | ~ under |
| fe_084 cupcake red velvet | ~ LEAK glow | OK |
| fe_108 duck colorful | OK | OK |
| fe_114/115 flowers | OK | ~ texture drift |
| fe_128 gray bird red | X MELT species+SCALE | OK (cardinal) |
| fe_140 horse brown | OK | X NOED |
| fe_141 iguana green | ~ LEAK white blob | OK |
| fe_199 cake red velvet | ~ layers chalky | OK |

Pattern: SD3 LEAK/MELT; FLUX clean or NOED. Backlit/white-target = both under-edit.

## T5 same_color_material (56) — best family overall, SD3 > FLUX on humans

Severe only:
- fe_038/039 cat&dog lego/bronze: SD3 X INST (cat untouched); FLUX OK-ish both.
- fe_080/081 corgi lego/wood: both X HALF (head-only transform, fur body).
- fe_093 dog lego: SD3 half-hearted; FLUX real lego but barely dog-shaped.
- fe_128-style species drift: fe_097 wooden dog -> labrador shape (~).
- fe_130 golden bird: SD3 SCALE shrink, FLUX SCALE inflate.
- fe_146 japanese castle lego: SD3 not-lego + scene change; FLUX X surreal (orange moon).
- fe_148 bronze kick: SD3 X material fail + panda face; FLUX ~ float.
- fe_150 marble kick: both ~ pose destroyed.
- fe_157 kid sculpture: FLUX X black silhouette.
- fe_152/153/171-176 statues: SD3 all good; FLUX ~ BGREP (cave->hallway on 175) + face melt.
- fe_193/194 parrots origami/sculpture: both X INST (right only).
- fe_196/197 penguins: SD3 X INST (left only); FLUX ~ both-ish.
- fe_211 puppies puppets: SD3 OK; FLUX X pink felt monsters.
- fe_221 rocks colorful blocks: both X INST (1-2 rocks only).
- fe_252 tiger crochet: SD3 toy-ish OK; FLUX good crochet texture.
- fe_266-273 bulldog origami series: SD3 all good; FLUX vague shapes + X same grey smudge artifact on marble bg every case.

## Cross-family mechanism buckets (fix priorities)

1. INST plural under-count — T1 192/195, T5 038/039/193/194/196/197/221 (+T2 036 berries). Both backends, 3 families. => instance-aware support proposal. BIGGEST systematic bucket.
2. SD3 text BLANK/CCOL/LDROP — ~15 cases. Known mechanism (preserve-first collapse, size-gated conditioning).
3. FLUX GHOST text overlay — ~10 cases (erase half missing: writes without removing).
4. FLUX NOED under-edit — T2 insertions + T4 hard recolor + 246. => edit-budget/underfire.
5. SD3 LEAK halo — 015/016/084/141. => guidance field spill outside mask.
6. FLUX BGREP — T5 statues/castle/bulldog smudge, T3 grid paper. => preserve field too weak outside edit region on FLUX.
7. HALF conversion — corgi 080/081. => support mask covers only high-attention part of subject.
8. MELT/SCALE — butterflies, bird 128, faces. => structure preservation under strong material/color change.
9. RESID replace — 046 (+T1 iguana hat). => replace must cover source-object region (removed-token attention).
