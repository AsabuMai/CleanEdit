# Phase2 Clean/Velocity Ablation Full Summary

Scope: full Phase2 T1-T5 ablation over 15 tasks and seeds 10/11/12.

| method | n | outside_l1 | bg_psnr | bg_lpips_x100 | bg_ssim_luma | dino_source | clip_t | local_clip_t | edit_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support_v3_controller_rmsgap | 45 | 0.0279 | 28.9279 | 5.3014 | 0.7999 | 0.8847 | 0.2770 | 0.2788 | 0.0333 |
| support_v3_controller_rmsgap_ablate_legacy_conversion | 45 | 0.0283 | 28.9617 | 7.1541 | 0.7982 | 0.8427 | 0.2812 | 0.2817 | 0.0361 |
| support_v3_controller_rmsgap_ablate_rf_diff_editfield | 45 | 0.0280 | 28.8412 | 5.2477 | 0.7998 | 0.8578 | 0.2806 | 0.2820 | 0.0370 |
| support_v3_controller_rmsgap_ablate_support_attn_clean | 45 | 0.0282 | 28.7670 | 5.4990 | 0.7989 | 0.8571 | 0.2701 | 0.2659 | 0.0235 |
| support_v3_controller_rmsgap_ablate_support_attn_velocity | 45 | 0.0282 | 28.7721 | 5.4859 | 0.7990 | 0.8668 | 0.2714 | 0.2682 | 0.0242 |
