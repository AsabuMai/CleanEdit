# Efficiency Context Table (2026-06-11)

Audit status: `complete` (`experiments/support_v3_2026-06-02/efficiency_context_2026-06-11_audit.json`).

Runtime and peak GPU memory are averaged only over runs whose producer recorded those fields. Missing external-baseline runtime/peak fields are disclosed by the availability columns rather than imputed.

| label | backbone | n | nfe | resolution | runtime_s_mean | runtime_s_available | peak_gpu_gb_mean | peak_gpu_gb_available | offload_recorded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF reconstruction | SD3 | 45 | T_steps=28,n_max=24 | 336x512,512x336,512x352,512x432 | 31.33 | 45/45 | 12.24 | 45/45 | not_recorded |
| Direct target | SD3 | 45 | T_steps=28,n_max=24 | 336x512,512x336,512x352,512x432 | 32.08 | 45/45 | 12.24 | 45/45 | not_recorded |
| Generic support | SD3 | 45 | T_steps=28,n_max=24 | 336x512,512x336,512x352,512x432 | 50.18 | 45/45 | 13.25 | 45/45 | not_recorded |
| Fixed CleanEdit-SD3 | SD3 | 45 | T_steps=28,n_max=24 | 336x512,512x336,512x352,512x432 | 68.35 | 45/45 | 13.45 | 45/45 | not_recorded |
| CleanEdit-SD3 | SD3 | 45 | T_steps=28,n_max=24 | 336x512,512x336,512x352,512x432 | 56.89 | 45/45 | 13.35 | 45/45 | not_recorded |
| FlowEdit-SD3 | SD3 | 45 | T_steps=28,n_max=24 | 336x512,512x336,512x352,512x432 | not_recorded | 0/45 | not_recorded | 0/45 | not_recorded |
| FlowAlign-SD3 | SD3 | 45 | 33 | 336x512,512x336,512x352,512x432 | not_recorded | 0/45 | not_recorded | 0/45 | not_recorded |
| SplitFlow-SD3 | SD3 | 45 | T_steps=50,n_max=33 | 336x512,512x336,512x352,512x432 | not_recorded | 0/45 | not_recorded | 0/45 | not_recorded |
| FireFlow-FLUX/context | FLUX/context | 45 | 8 | 336x512,512x336,512x352,512x432 | not_recorded | 0/45 | not_recorded | 0/45 | yes |
| RF-Solver-Edit-FLUX/context | FLUX/context | 45 | 15 | 336x512,512x336,512x352,512x432 | not_recorded | 0/45 | not_recorded | 0/45 | yes |
| ReFlex-FLUX/context | FLUX/context | 45 | 28 | 341x512,342x512,512x341,512x342,512x356,512x440 | not_recorded | 0/45 | not_recorded | 0/45 | not_recorded |

Interpretation: this is a cost/context table, not an efficiency claim. Native/context rows use different backbones and interfaces.
