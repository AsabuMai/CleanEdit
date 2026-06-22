# FlowEdit-135 top-10 qualitative candidates

| # | task | type | why selected | ours local/bgLPIPS | nearest visual competitors |
|---:|---|---|---|---|---|
| 1 | `fe_094_dog_6_red_top_hat` | T1 / red top hat | clear accessory edit; grass/background mostly fixed | C=0.257, LP=0.0522 | edit: FlowEdit-SD3 C=0.281; bg: OT-RF enhanced LP=0.0500 |
| 2 | `fe_195_parrots2_1_crown` | T1 / parrots with crowns | accessory edit with stronger object alignment than iguana | C=0.295, LP=0.0487 | edit: RF-Solver C=0.312; bg: InstructPix2Pix LP=0.0960 |
| 3 | `fe_031_cake_1_berries` | T2 / berries on cake | localized insertion; almost no table/background drift | C=0.291, LP=0.0005 | edit: RF-Solver C=0.299; bg: SAM-Flow-FLUX LP=0.0023 |
| 4 | `fe_201_pizza_1_pineapple_ham` | T2 / pineapple + ham pizza | ingredient insertion without changing board/table | C=0.284, LP=0.0004 | edit: RF-Solver C=0.309; bg: SAM-Flow-FLUX LP=0.0030 |
| 5 | `fe_116_free_wifi_1_free_beer` | T3 / FREE BEER sign | text surface edit; board/table remain stable | C=0.243, LP=0.0001 | edit: FireFlow C=0.259; bg: SAM-Flow-FLUX LP=0.0005 |
| 6 | `fe_244_stop_sticker_1_cvpr` | T3 / CVPR sticker | decal/text replacement; very clean non-edit region | C=0.365, LP=0.0006 | edit: RF-Solver C=0.345; bg: SAM-Flow-SD3 LP=0.0038 |
| 7 | `fe_025_bus_3_pink_van` | T4 / pink van | visible recolor; house/garage preserved | C=0.266, LP=0.0047 | edit: LEDITS++ C=0.297; bg: SAM-Flow-FLUX LP=0.0321 |
| 8 | `fe_084_cupcake_2_red_velvet` | T4 / red velvet cupcake | local recolor/style change with stable plate/background | C=0.244, LP=0.0032 | edit: FlowEdit-FLUX C=0.288; bg: SAM-Flow-FLUX LP=0.0228 |
| 9 | `fe_175_meditation1_4_golden_statue` | T5 / golden statue | material replacement; ours stronger than many baselines on edit score | C=0.242, LP=0.0007 | edit: FlowEdit-FLUX C=0.245; bg: SAM-Flow-FLUX LP=0.0045 |
| 10 | `fe_193_parrots_2_origami` | T5 / origami parrots | material/style edit; baselines often disturb surroundings | C=0.310, LP=0.0013 | edit: ReFLEX C=0.320; bg: SAM-Flow-FLUX LP=0.0617 |

Reserve candidates:

- `fe_142_iguana_2_blue_lizard_top_hat`: good preservation but top hat floats slightly
- `fe_095_dog_7_jeweled_crown`: stable, but repeats the dog source
- `fe_055_cat_7_wooden_sculpture`: strong T5 backup
- `fe_127_gray_bird_1_origami_bird`: clean T5 origami backup
- `fe_198_piece_of_cake_1_cherry_on_top`: clean T2 backup, but cherry is small

Generated figures:

- `experiments/flowedit135_fixedmask_metrics_20260621/paper_top10_comparison_grid.jpg`
- `experiments/flowedit135_fixedmask_metrics_20260621/paper_top10_all_baselines_audit.jpg`