# SD3 / FLUX Backend Correspondence

This note records the project-level rule for porting the paper SD3 CleanEdit
implementation to FLUX. SamFlow is used here only as an interface pattern:
one task interface, one algorithmic state, and thin backend adapters for each
model family.

## SamFlow Pattern

SamFlow keeps the experiment contract identical across SD3 and FLUX:

- `scripts/run_image.py` receives the same task fields: source image, source
  prompt, target prompt, source tokens, target tokens, unchanged tokens, seed,
  and output location.
- `--mode sd3` and `--mode flux` choose the backend runner.
- Both backend runners use the same algorithm variables: source latent, target
  scout latent, source/target prompt embeddings, token-derived edit map, core
  map, ring map, accumulated visible gate, source anchor, model state, visible
  state, and velocity delta.
- Backend differences are confined to encoding, prompt APIs, transformer
  forward calls, scheduler sigma/timestep conventions, and latent layout.
  SD3 works in BCHW latent maps; FLUX works in packed image tokens and unpacks
  before VAE decoding.

The important lesson is not to copy SamFlow's algorithm. The useful part is the
correspondence boundary: the algorithm sees the same conceptual tensors, while
each backend supplies them in its native layout.

## Our CleanEdit Contract

For the paper claim, the SD3 implementation remains the canonical mathematical
definition. The FLUX version should be described as the same CleanEdit objective
and update after applying a layout adapter from FLUX packed tokens to BCHW maps.

The shared per-step equation is:

```text
v_total = v_src + v_rec + v_edit
z_{next} = z_t + (sigma_next - sigma_t) * v_total
```

where:

- `v_src` is the source-prompt rectified-flow velocity in the native backend
  latent layout.
- `v_rec` is the preserve/reconstruction correction from
  `reconstruction_velocity_surrogate_total`.
- `v_edit` is the edit correction from `editing_velocity_surrogate_total`,
  plus the same optional local target, region transport, recolor projection,
  adaptive projection, and removal controller terms when enabled.
- `M_edit`, `M_preserve`, `edit_gate`, `preserve_gate`, and optional `core_gate`
  define the same spatial regions as in the SD3 paper code.

## Current Project Mapping

The project now has the intended SamFlow-style split:

- `scripts/run_dece_image.py` is the shared CLI wrapper. It takes common task
  and controller arguments, then dispatches to `run_edit_sd3.py` or
  `run_edit_flux.py` via `--backend sd3|flux`.
- `sd3_hrec.py` is the canonical paper implementation. It still contains the
  SD3 equations inline and calls the shared energy functions directly.
- `dece_core.py` is the backend-neutral CleanEdit step extracted from the SD3
  equations. It contains no SD3 or FLUX model API calls.
- `flux/dece_flux_adapter.py` is the FLUX adapter. It converts FLUX packed
  tokens to BCHW maps for the shared energy functions and converts the resulting
  correction back to packed tokens.
- `flux/flux_hrec.py` is the FLUX backend runner. Its `dece_rf_flux` path calls
  `FluxDeceAdapter.compute_step(build_flux_dece_config(args), ...)` and uses
  `core_out.v_total` for the RF update.

## What Must Stay Identical

The following are mathematical interface invariants. If one side changes, the
other side must be audited:

- Same source/target clean estimate definitions:
  `x0_src = predict_x0(z_t, v_src, sigma_t)` and
  `x0_tar = predict_x0(z_t, v_tar, sigma_t)`.
- Same base edit velocity:
  `base_edit_velocity = v_tar - v_src_edit`.
- Same reconstruction surrogate inputs:
  `x0_src`, `x_src`, `M_preserve`, `struct_guidance_scale`,
  `linear_path_t_min`.
- Same edit surrogate inputs:
  `base_edit_velocity`, `x0_tar`, `x0_src`, `M_edit`, target/source feature
  maps when enabled, and the five edit weights.
- Same optional controller semantics for adaptive clean control, local target
  formation, region target transport, recolor clean projection, and removal.
- Same RF update sign convention:
  `(sigma_next - sigma_t) * v_total`.

## What May Differ By Backend

These differences are implementation details, not new mathematical claims:

- Prompt encoding and CFG mechanics.
- Scheduler timestep representation, as long as the scalar `sigma_t` passed to
  the CleanEdit core has the same RF meaning.
- Latent layout:
  SD3 uses BCHW maps; FLUX uses packed image tokens. The adapter must be a pure
  reshape/transposition for tensors that represent the same latent field.
- Image-id/text-id handling required by FLUX transformer calls.
- Attention/token grounding extraction, provided it produces the same semantic
  role masks expected by the CleanEdit step.

## Paper-Safe Wording

The defensible claim is:

> We use the SD3 implementation as the canonical CleanEdit formulation. The FLUX
> variant preserves the same CleanEdit velocity decomposition, energy surrogates,
> masks, schedules, and RF update, while replacing only backend-specific model
> calls and latent layout handling through a packed-token adapter.

Avoid claiming that every numerical trajectory is identical across SD3 and
FLUX. The backbones, prompt encoders, schedulers, and latent coordinate systems
are different, so equality should be stated at the level of objective,
controller terms, and update form.
