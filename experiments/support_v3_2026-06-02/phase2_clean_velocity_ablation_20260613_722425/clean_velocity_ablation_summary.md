# Phase2 Clean/Velocity Ablation Summary

Scope: smoke ablation over the selected TASKS and SEEDS from the sbatch environment.

| method | n | outside_l1 | bg_psnr | bg_lpips_x100 | bg_ssim_luma | dino_source | clip_t | local_clip_t | edit_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support_v3_controller_rmsgap | 5 | 0.0351 | 27.2471 | 6.2690 | 0.7416 | 0.9055 | 0.2846 | 0.2757 | 0.0475 |
| support_v3_controller_rmsgap_ablate_legacy_conversion | 5 | 0.0358 | 27.3869 | 9.2484 | 0.7365 | 0.8800 | 0.2886 | 0.2714 | 0.0480 |
| support_v3_controller_rmsgap_ablate_rf_diff_editfield | 5 | 0.0351 | 27.2654 | 6.2736 | 0.7416 | 0.8693 | 0.2826 | 0.2662 | 0.0459 |
| support_v3_controller_rmsgap_ablate_support_attn_clean | 5 | 0.0367 | 26.8595 | 7.2124 | 0.7381 | 0.8656 | 0.2672 | 0.2418 | 0.0272 |
| support_v3_controller_rmsgap_ablate_support_attn_velocity | 5 | 0.0367 | 26.8522 | 7.2038 | 0.7382 | 0.9002 | 0.2718 | 0.2492 | 0.0280 |
