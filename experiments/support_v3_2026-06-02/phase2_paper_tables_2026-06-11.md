# Phase2 Paper Tables (2026-06-11)

Current project lock: Phase2 means T1-T5 with three source cases per family. It is the main experiment stage, not a separate task category.

Audit status: `complete` (`experiments/support_v3_2026-06-02/phase2_tables_audit_2026-06-11.json`).

## Table 1: Phase2 T1-T5 Main Effect

Scope: 15 task cases x seeds 10/11/12 = 45 rows per method.
Metrics: Non-edit-region mean absolute error (Non-edit MAE) measures the average pixel deviation between the edited output and source image over the fixed non-edit evaluation region. BG metrics use M_ne = 1 - Dilate(M_edit) to exclude the edit boundary. BG-LPIPS is reported as x100. CLIP-T is full-image target prompt similarity; Local CLIP-T is edit-crop similarity to a task-specific local target phrase. Inside L1 is descriptive and reported only in the supplement block.

Formula: NonEdit-MAE = |M_ne|^{-1} || M_ne \odot (\hat{x} - x_src) ||_1.

| label | n | Non-edit MAE ↓ | BG-PSNR ↑ | BG-LPIPS×100 ↓ | BG-SSIM-luma ↑ | DINO-source ↑ | CLIP-T ↑ | Local CLIP-T ↑ | edit_score ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF reconstruction | 45 | 0.0487 | 23.1385 | 27.9459 | 0.6995 | 0.6985 | 0.2561 | 0.2087 | -0.0179 |
| Direct target | 45 | 0.0729 | 18.8534 | 35.4833 | 0.6499 | 0.6077 | 0.2790 | 0.2313 | 0.0096 |
| Generic support | 45 | 0.0351 | 24.2828 | 8.9691 | 0.7489 | 0.8998 | 0.2673 | 0.2382 | 0.0026 |
| CleanEdit-SD3 | 45 | 0.0281 | 28.8510 | 5.4470 | 0.7996 | 0.8619 | 0.2788 | 0.2795 | 0.0379 |

## Table 2a: Same-Backbone SD3 Common Subset

| label | n | Non-edit MAE ↓ | BG-PSNR ↑ | BG-LPIPS×100 ↓ | BG-SSIM-luma ↑ | DINO-source ↑ | CLIP-T ↑ | Local CLIP-T ↑ | edit_score ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct target | 45 | 0.0729 | 18.8534 | 35.4833 | 0.6499 | 0.6077 | 0.2790 | 0.2313 | 0.0096 |
| Generic support | 45 | 0.0351 | 24.2828 | 8.9691 | 0.7489 | 0.8998 | 0.2673 | 0.2382 | 0.0026 |
| FlowEdit-SD3 | 45 | 0.1614 | 13.8071 | 46.1008 | 0.5333 | 0.4871 | 0.2962 | 0.2511 | 0.0434 |
| SplitFlow-SD3 | 45 | 0.0774 | 19.3834 | 26.5071 | 0.6473 | 0.6281 | 0.2958 | 0.2765 | 0.0387 |
| Sam-Flow-SD3 | 45 | 0.0516 | 21.5160 | 14.9820 | 0.7115 | 0.7054 | 0.2938 | 0.2865 | 0.0397 |
| Fixed CleanEdit-SD3 | 45 | 0.0285 | 28.7194 | 5.8399 | 0.7991 | 0.8624 | 0.2828 | 0.2828 | 0.0382 |
| CleanEdit-SD3 | 45 | 0.0281 | 28.8510 | 5.4470 | 0.7996 | 0.8619 | 0.2788 | 0.2795 | 0.0379 |

## Table 2b: Native Preservation-Aware RF / FLUX Context

Scope: 14 task cases x seeds 10/11/12 = 42 rows per method. The white_bowl_orange_tabletop case is excluded from this native-FLUX comparison (all methods) because the small free-standing orange is below the FLUX generation floor for every native-FLUX method; Tables 1 and 2a retain all 15 cases.

| label | n | Non-edit MAE ↓ | BG-PSNR ↑ | BG-LPIPS×100 ↓ | BG-SSIM-luma ↑ | DINO-source ↑ | CLIP-T ↑ | Local CLIP-T ↑ | edit_score ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FireFlow-FLUX/context | 42 | 0.0688 | 20.8760 | 31.6467 | 0.6352 | 0.6841 | 0.2849 | 0.2715 | 0.0213 |
| RF-Solver-Edit-FLUX/context | 42 | 0.0559 | 22.1177 | 25.9563 | 0.6672 | 0.7501 | 0.2956 | 0.2740 | 0.0286 |
| ReFlex-FLUX/context | 42 | 0.0861 | 18.6825 | 34.7778 | 0.7313 | 0.7108 | 0.2812 | 0.2773 | 0.0324 |
| Sam-Flow-FLUX/context | 42 | 0.0439 | 22.8237 | 10.2909 | 0.7319 | 0.7977 | 0.2883 | 0.2756 | 0.0336 |
| CleanEdit-FLUX/context | 42 | 0.0024 | 48.7449 | 0.4320 | 0.9965 | 0.8872 | 0.2797 | 0.2715 | 0.0370 |

## Supplement: Phase2 Family Breakdown

| family | label | n | Non-edit MAE ↓ | BG-PSNR ↑ | BG-LPIPS×100 ↓ | BG-SSIM-luma ↑ | DINO-source ↑ | CLIP-T ↑ | Local CLIP-T ↑ | edit_score ↑ | inside_l1 descriptive | BG-DINO ↑ | Source SSIM-luma ↑ | CLIP delta ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_attached_accessory | RF reconstruction | 9 | 0.0848 | 18.4756 | 37.4090 | 0.4626 | 0.5392 | 0.2345 | 0.1934 | -0.0298 | 0.0843 | 0.6134 | 0.4443 | -0.0298 |
| T1_attached_accessory | Direct target | 9 | 0.0947 | 17.4866 | 36.7040 | 0.4499 | 0.5683 | 0.2524 | 0.2301 | -0.0034 | 0.1364 | 0.6656 | 0.4277 | -0.0034 |
| T1_attached_accessory | Generic support | 9 | 0.0582 | 20.3301 | 10.5314 | 0.5469 | 0.8935 | 0.2365 | 0.1974 | -0.0190 | 0.0490 | 0.8991 | 0.5339 | -0.0190 |
| T1_attached_accessory | CleanEdit-SD3 | 9 | 0.0566 | 20.8845 | 8.0624 | 0.5501 | 0.9230 | 0.2849 | 0.2887 | 0.0686 | 0.1333 | 0.9792 | 0.5246 | 0.0686 |
| T2_container_insertion | RF reconstruction | 9 | 0.0484 | 22.5738 | 31.4355 | 0.6603 | 0.6312 | 0.2469 | 0.1856 | -0.0196 | 0.0427 | 0.6768 | 0.6656 | -0.0196 |
| T2_container_insertion | Direct target | 9 | 0.0733 | 18.6271 | 45.3977 | 0.5976 | 0.4896 | 0.2833 | 0.1910 | 0.0112 | 0.0769 | 0.5186 | 0.5984 | 0.0112 |
| T2_container_insertion | Generic support | 9 | 0.0388 | 22.7034 | 10.5449 | 0.7171 | 0.7830 | 0.2860 | 0.2531 | 0.0164 | 0.0779 | 0.8582 | 0.7132 | 0.0164 |
| T2_container_insertion | CleanEdit-SD3 | 9 | 0.0358 | 23.1665 | 9.0453 | 0.7296 | 0.7071 | 0.2910 | 0.2612 | 0.0246 | 0.0902 | 0.8200 | 0.7229 | 0.0246 |
| T3_surface_decal | RF reconstruction | 9 | 0.0256 | 25.9382 | 15.6148 | 0.8821 | 0.8716 | 0.2815 | 0.2143 | -0.0319 | 0.0222 | 0.8396 | 0.8808 | -0.0319 |
| T3_surface_decal | Direct target | 9 | 0.0467 | 20.3145 | 24.1948 | 0.8378 | 0.7506 | 0.2823 | 0.2567 | 0.0008 | 0.0830 | 0.7430 | 0.8277 | 0.0008 |
| T3_surface_decal | Generic support | 9 | 0.0239 | 25.4575 | 6.5656 | 0.8836 | 0.9567 | 0.2812 | 0.2100 | -0.0175 | 0.0195 | 0.9538 | 0.8828 | -0.0175 |
| T3_surface_decal | CleanEdit-SD3 | 9 | 0.0164 | 27.5771 | 3.9720 | 0.9061 | 0.7778 | 0.2834 | 0.2980 | 0.0647 | 0.1234 | 0.9420 | 0.8939 | 0.0647 |
| T4_local_recolor | RF reconstruction | 9 | 0.0427 | 24.5430 | 25.0587 | 0.7730 | 0.7508 | 0.2650 | 0.2084 | -0.0146 | 0.0690 | 0.7484 | 0.7672 | -0.0146 |
| T4_local_recolor | Direct target | 9 | 0.0596 | 20.6336 | 27.0351 | 0.7577 | 0.7330 | 0.2915 | 0.2246 | 0.0192 | 0.1290 | 0.7601 | 0.7407 | 0.0192 |
| T4_local_recolor | Generic support | 9 | 0.0297 | 27.6107 | 5.8230 | 0.8184 | 0.9220 | 0.2866 | 0.2876 | 0.0352 | 0.3855 | 0.9510 | 0.8290 | 0.0352 |
| T4_local_recolor | CleanEdit-SD3 | 9 | 0.0295 | 27.8103 | 5.5689 | 0.8184 | 0.9190 | 0.2873 | 0.2865 | 0.0382 | 0.3935 | 0.9468 | 0.8261 | 0.0382 |
| T5_same_color_material | RF reconstruction | 9 | 0.0422 | 24.1617 | 30.2114 | 0.7195 | 0.6998 | 0.2525 | 0.2417 | 0.0065 | 0.0388 | 0.7203 | 0.7208 | 0.0065 |
| T5_same_color_material | Direct target | 9 | 0.0904 | 17.2053 | 44.0850 | 0.6063 | 0.4969 | 0.2857 | 0.2540 | 0.0201 | 0.1724 | 0.5466 | 0.5645 | 0.0201 |
| T5_same_color_material | Generic support | 9 | 0.0249 | 25.3122 | 11.3807 | 0.7787 | 0.9438 | 0.2461 | 0.2428 | -0.0021 | 0.0236 | 0.9449 | 0.7761 | -0.0021 |
| T5_same_color_material | CleanEdit-SD3 | 9 | 0.0024 | 44.8166 | 0.5863 | 0.9936 | 0.9827 | 0.2473 | 0.2631 | -0.0065 | 0.0252 | 0.9970 | 0.9694 | -0.0065 |

Do not report the old five-canonical-case Core-5 tables as current evidence. E5 removal remains a separate boundary probe.
