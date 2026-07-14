# Core-5 Mask Sensitivity Sweep

Audit status: `complete` (`experiments/support_v3_2026-06-02/mask_sensitivity_core5_audit.json`).

Evaluation masks are eroded/base/dilated with a 15x15 min/max filter. Ranking uses mean outside-mask L1, ascending.

Full order stable: `True`. Top method stable: `True`.

| variant | label | n | outside_l1 | inside_l1 | source_ssim_luma |
| --- | --- | --- | --- | --- | --- |
| eroded | CleanEdit-SD3 | 15 | 0.0404 | 0.1026 | 0.7094 |
| eroded | Generic support | 15 | 0.0480 | 0.0782 | 0.6757 |
| eroded | RF reconstruction | 15 | 0.0707 | 0.0517 | 0.5786 |
| eroded | Direct target | 15 | 0.0918 | 0.0838 | 0.5141 |
| base | CleanEdit-SD3 | 15 | 0.0393 | 0.0972 | 0.7094 |
| base | Generic support | 15 | 0.0470 | 0.0791 | 0.6757 |
| base | RF reconstruction | 15 | 0.0708 | 0.0564 | 0.5786 |
| base | Direct target | 15 | 0.0911 | 0.0912 | 0.5141 |
| dilated | CleanEdit-SD3 | 15 | 0.0388 | 0.0894 | 0.7094 |
| dilated | Generic support | 15 | 0.0464 | 0.0767 | 0.6757 |
| dilated | RF reconstruction | 15 | 0.0708 | 0.0590 | 0.5786 |
| dilated | Direct target | 15 | 0.0899 | 0.0984 | 0.5141 |

Rankings:
- `eroded`: CleanEdit-SD3 > Generic support > RF reconstruction > Direct target
- `base`: CleanEdit-SD3 > Generic support > RF reconstruction > Direct target
- `dilated`: CleanEdit-SD3 > Generic support > RF reconstruction > Direct target
