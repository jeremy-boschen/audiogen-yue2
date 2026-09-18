# Matching the ComfyUI output

> Superseded investigation notes. Current measured results and corrected causes
> are in [PARITY_PROGRESS.md](PARITY_PROGRESS.md). The original album riff and
> full recorded continuation chain now reproduce exactly without ComfyUI packages.
> See [PARITY_ACCEPTANCE.md](PARITY_ACCEPTANCE.md) for remaining gates. The older
> numerical comparisons below are not current acceptance proof.

The album in [audiogen-comfyui](https://github.com/jeremy-boschen/audiogen-comfyui)
was rendered by the ComfyUI FL-YuE2 node pack. This stack uses the official
library. **They are different performers**, and the difference was bisected
op-by-op on an identical 766-token prefix rather than guessed at.

Four causes, in execution order:

1. **RMSNorm rounding.** ComfyUI's `operations.RMSNorm` calls `F.rms_norm` (one
   rounding). The library hand-rolls
   `x * rsqrt(x.float().pow(2).mean(-1)+eps).to(x.dtype) * w`, casting to bf16
   *before* the weight multiply (three roundings). This is the first divergent op
   in the whole network. Matching it makes everything through `v_proj` bit-exact.
2. **Fused `comfy_kitchen.rms_rope_split_half`** -- q_norm, k_norm and rotary in
   one kernel with a 2x2 rotation, against the library's four separate ops and
   split-half rotary. Matching it makes q/k/v bit-exact.
3. **Chunked attention.** The pack splits into 256-token blocks with explicit
   `-inf` masks; the library does one full-sequence SDPA. Matching it cuts the
   attention residual 16x but not to zero.
4. **torch version.** 2.10.0 here against 2.14.0 in the ComfyUI venv. Proven with
   a standalone probe using no model code at all: same q/k/v, same
   `scaled_dot_product_attention`, 88.42% bit-exact, max 0.00390625 -- *exactly*
   the in-model residual. **This is not a code difference and no amount of porting
   closes it.**

Causes 1-3 are portable as opt-in shims. Cause 4 is a version pin.

## What that means in practice

Six torch versions produce **four distinct songs** from one seed: 2.11.0, 2.12.0
and 2.12.1 are byte-identical to each other; 2.10.0, 2.13.0 and 2.14.0 each stand
alone. So "match ComfyUI" means matching *both* the three shims and the torch
version, and the torch version alone re-rolls the take.

A caution measured the hard way: **partial fixes can make the end-to-end logit
difference worse.** Applying shims 1+2 without 3 gave mean 0.0209 against a 0.0169
baseline, because errors that previously cancelled no longer did. Only the full
set improved it, to 0.0097. Do not ship a partial match and call it progress.

## What survives regardless

The supplied score pins structure -- key, tempo, bars, chord changes, vocal
entries -- and nothing numerical moves it. Everything below it (sustain,
articulation, how hard drums land, timbre) is sampled and re-rolls entirely.

## Do not measure divergence by dividing a token index by 25

The acoustic stage attends over the whole sequence, so a token that first differs
at 35.04s changes the latent at frame 0. Verified: latent frame 0 differs in every
comparison, including a take whose tokens agreed for 876 steps. Measure the
waveform, not the token index.
