# FlowEdit-135 target-aligned top-10 candidates

| # | task | type | rationale | ours local/bgLPIPS | best baseline edit / best baseline bg |
|---:|---|---|---|---|---|
| 1 | `fe_094_dog_6_red_top_hat` | T1 / red top hat | target is clear; dog/background mostly preserved | C=0.257, LP=0.0522 | FlowEdit-SD3 C=0.281; OT-RF enhanced LP=0.0500 |
| 2 | `fe_192_parrots_1_top_hat` | T1 / one parrot top hat | single-accessory target matches better than two-crown case | C=0.265, LP=0.0322 | FireFlow C=0.294; InstructPix2Pix LP=0.0462 |
| 3 | `fe_031_cake_1_berries` | T2 / berries on cake | target insertion is visible and localized | C=0.291, LP=0.0005 | RF-Solver C=0.299; SAM-Flow-FLUX LP=0.0023 |
| 4 | `fe_201_pizza_1_pineapple_ham` | T2 / pineapple + ham pizza | target toppings are clear; board/table stable | C=0.284, LP=0.0004 | RF-Solver C=0.309; SAM-Flow-FLUX LP=0.0030 |
| 5 | `fe_116_free_wifi_1_free_beer` | T3 / FREE BEER sign | text target is readable; surrounding board/table stable | C=0.243, LP=0.0001 | FireFlow C=0.259; SAM-Flow-FLUX LP=0.0005 |
| 6 | `fe_244_stop_sticker_1_cvpr` | T3 / CVPR sticker | target text is readable; non-edit background clean | C=0.365, LP=0.0006 | RF-Solver C=0.345; SAM-Flow-SD3 LP=0.0038 |
| 7 | `fe_025_bus_3_pink_van` | T4 / pink van | target color/object identity clear; house preserved | C=0.266, LP=0.0047 | LEDITS++ C=0.297; SAM-Flow-FLUX LP=0.0321 |
| 8 | `fe_084_cupcake_2_red_velvet` | T4 / red velvet cupcake | target red-velvet cue visible; background minimal | C=0.244, LP=0.0032 | FlowEdit-FLUX C=0.288; SAM-Flow-FLUX LP=0.0228 |
| 9 | `fe_055_cat_7_wooden_sculpture` | T5 / wooden kitten sculpture | material target is clearer than origami parrots | C=0.308, LP=0.0009 | SAM-Flow-SD3 C=0.340; SAM-Flow-FLUX LP=0.0048 |
| 10 | `fe_175_meditation1_4_golden_statue` | T5 / golden statue | material target strong; cave/background preserved | C=0.242, LP=0.0007 | FlowEdit-FLUX C=0.245; SAM-Flow-FLUX LP=0.0045 |

Demoted from previous visual-first list:

- `fe_195_parrots2_1_crown`: target asks both parrots have crowns; ours reads closer to one dominant crown
- `fe_193_parrots_2_origami`: good preservation, but origami transformation is not as unambiguous as desired
- `fe_142_iguana_2_blue_lizard_top_hat`: hat floats slightly; target alignment less clean
- `fe_198_piece_of_cake_1_cherry_on_top`: clean, but edit is very small for a main qualitative row