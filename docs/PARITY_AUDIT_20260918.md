# Follow-up parity audit, 2026-09-18

Baseline01 parity remains unresolved. This audit qualifies conclusions in
`DIFF_VS_COMFYUI.md`; it does not replace the historical measurements.

## Measurement reproduced

Re-ran the existing `tf3.py` teacher-forcing probe with `SHIMS=norm_c` against
`greedy45.json`: 27/1125 mismatches, first at zero-based step 126, median
score margin -0.0625, worst -0.1250. These match the recorded result.

The probe allocated `len(prefix)+len(their)+8` KV slots; production sampling
allocates `len(prefix)+sampling.max_tokens`. Here the target length and
sampling budget are both 1125. A second run removed only the extra eight slots.
All reported metrics remained identical. This eliminates the padding as a
cause of the reported mismatch count in this configuration; it does not prove
tensor identity.

Evidence: `../../audiogen/output/parity_cache_capacity_20260918/` contains both
probe versions, input SHA-256 hashes, the original result transcribed from tool
output, and the exact-capacity run's captured log. No model code, environment,
or ComfyUI server settings were changed.

## Conclusions the existing evidence does not establish

- A higher mismatch count after enabling Metal attention does not rule out
  that kernel as a contributor. Other differences can cancel or amplify its
  effect. The claim that the stock path is numerically between the two paths
  needs tensor measurements; token agreement rates do not establish it.
- Earlier first-token disagreement does not measure perturbation magnitude.
  It depends on token ranking margins and the direction of the perturbation.
- Teacher forcing fixes token history, not hidden states or KV-cache contents.
  Prefill differences can persist into every decode step.
- `tf3.py` measures `our_scores[target_token] - our_scores[our_argmax]`.
  Those are within-model, post-penalty margins, not cross-model logit errors.
  Thus the reported margins cannot establish that all cross-model differences
  are one BF16 ULP, or that all disagreements are caused by that rounding.
- Unchanged tokens establish unchanged decisions in the tested context, not
  numerical equivalence of an attention or normalization implementation.

## Next decisive experiment

Capture the actual ComfyUI runtime and engine on the same saved prefix, first
at prefill and then at the first decode call. Keep the existing configurations
fixed while measuring. Record actual function bindings and effective weights,
plus tensor dtype, shape and stride. Compare, in execution order:

1. Embedding and layer input.
2. Input RMSNorm and projected Q/K/V.
3. Q/K normalization and rotary outputs, positions and active KV cache.
4. Attention inputs/output, output projection, residual, MLP and layer output.
5. Final norm, raw logits and post-penalty scores.

Start with layer zero and locate the earliest unequal boundary. If attention
receives unequal inputs, move upstream. If it receives equal inputs, replay
both real attention implementations on those exact inputs, preserving layout
and dispatch settings. Use bitwise comparison plus absolute error statistics;
do not infer tensor equality from argmax agreement.

Validate capture against an uncaptured control so that hooks or device copies
do not silently change the reference. Retain full-prefix prefill information:
capturing only the last query or only step 126 can miss the origin of cache
divergence. Extend to step 126 only after the earlier boundaries are understood.

After AR parity, test NAR on identical semantic tokens, then VAE on identical
latents. AR agreement alone cannot establish audio parity.
