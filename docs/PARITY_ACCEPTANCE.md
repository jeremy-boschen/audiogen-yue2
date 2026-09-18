# Exact-output acceptance audit

Current export policy: production now uses the engine's native 24-bit, 48 kHz
FLAC writer. The 16-bit ComfyUI PCM comparisons below describe the preserved
historical validation runs. New production files intentionally use a different
bit depth and encoder; those 16-bit file comparisons are not a current export gate.
Numerical generation still uses the same engine profile.

The subsequent engine-profile migration is also complete: see
[PROFILE_MIGRATION.md](PROFILE_MIGRATION.md) for the fresh pinned-package chain,
repeated fixtures and tensor checks. Numerical behavior now lives in engine-owned
profiles rather than application runtime patches. The original investigation
and evidence below are retained as history.

Status: **complete** on the matching hardware/runtime below. The original riff,
full four-stage lineage, two additional seeds and second input fixture reproduce
exactly in the fresh standalone environment. The corrected chain and validation
matrix each pass twice. Original-sampling tensor capture and same-input attention
replay also pass.
Historical investigation details and corrections are in `PARITY_PROGRESS.md`.
Evidence paths below are relative to `out/parity/`.

## Supported reference configuration

Apple M5 Pro, macOS 26.6.2, Python 3.13.15, torch 2.14.0 at git
`08187d9e0fba026dc8217405802ab5381dc88d90`, BF16 model / FP32 VAE,
`mtlflashattn==0.2.0`, `av==18.1.0`, album16-bit FLAC.
This is a matching-hardware/runtime claim, not cross-hardware bit identity.
Exact model/config/tokenizer hashes and reference source revisions are in
`frozen/manifest.json`; full native weight hashes accompany each generated take.
The reference launch uses `--use-pytorch-cross-attention --disable-smart-memory
--reserve-vram 8`. Production uses standalone operations and shared weights.

| Requirement | Measured evidence | Status |
|---|---|---|
| Freeze reference inputs, source, weights, runtime and launch | `frozen/manifest.json`, frozen diffs, trace `runtime.json`/`weights.json`, native numerical identities | Pass; `validated_source_state.json` binds chain_v3 to current source, pins and all frozen model files |
| Reproduce original discrepancy | `engine_normc_capture/result.json`:27/1125; corrected-input baseline still16/1125 | Pass |
| Real reference execution and instrumentation validity | Live greedy job/history; `reference_runtime_prefix/result.json` reproduces it with capture | Pass |
| Find earliest divergent operation on identical inputs | `norm_replay.json`, `attention_same_input_replay.json`, AR trace comparisons, `nar_stock_comparison.json` | Pass |
| Compare weights, embeddings, norms, Q/K/V, rotation, active cache, attention, residual/MLP, logits/scores | Captured boundary shapes/strides/dtypes/hashes and comparison rows; attention K/V inputs expose active cache contents | Pass; `final_sampled_trace_comparison.json` |
| Maintained corrections, source attribution and dependencies | `audiogen/parity.py`, `vendor/`, `continuation.py`, `audio.py`, hash lock, wheel contents | Pass |
| Teacher-forced and free-running AR, original sampling | Corrected-input teacher forcing0/1125; `engine_greedy_free`, `engine_sampled_free`; master294/294 | Pass |
| NAR on fixed tokens | `nar_engine_mtl/result.json`:72,000 latent values exact | Pass |
| VAE on fixed latents | `decode_engine_padding/result.json`:4,320,000 float32 values exact | Pass |
| Full original riff including encoded PCM | `album_riff_verified.json`; repeated `chain_v1_riff.json`, `chain_v2_riff.json` | Pass |
| Four-stage lineage with own generated predecessors | `chain_v2_verified/summary.json`, all four prefix/token/latent/PCM reports | Pass |
| Additional fixed seeds | Guitar778 and780 under `validation_v1`, two custom runs each | Pass |
| Second input fixture | Historical synth-style branch, seed777, two exact runs; historical PCM also verified | Pass |
| Repeated runs | `repeatability_matrix_verified.json`, `repeatability_chain_verified.json`; direct byte comparisons of prefixes/tokens/latents and decoded PCM across both runs | Pass |
| No ComfyUI installed or on production import path | `fresh_environment_no_comfy.json`; full chain rendered by that interpreter | Pass |
| Fresh repository environment | `fresh-venv` built with hash lock and pinned engine; full chain passes | Pass |
| Exact decoded PCM, matching sample rate/shape/duration | Comparisons decode both sides to integer PCM and require equality; container metadata excluded | Pass |

## Important evidence correction

The initial master prefix check stripped lyric whitespace and was invalid.
`chain_prefixes/master.json` must not be used as proof of reference equality.
The verifier now invokes the actual reference `runtime.make_plan`; all reports
in `chain_v2_verified/*_prefix.json` use that entrypoint. Master retains its
historical lyric newline. No numerical operation changed to correct that failure.

## Canonical checks

Render with `bin/render.py songs/burn_it_down_parity` and the three launch-equivalent
flags above. Compare with `diagnostics/check_chain.py`. The matrix runner is
`diagnostics/run_validation.py songs/parity_validation/* --out <new-directory>`;
it retains reference requests/job IDs/histories, custom commands/logs, actual
prefix comparisons and exact token/latent/PCM reports. Never substitute reference
intermediates into a custom acceptance run. Reference-only diagnostics use the
separate reference interpreter; production does not import its packages.

## Final measured audit

`final_sampled_{reference,engine}_capture/result.json` each reproduces all1125
original sampled tokens with capture enabled. `final_sampled_trace_comparison.json`
shows equal effective weights,1534 exact captured boundaries and two exact
selected-last-row final-norm comparisons; no reference boundary is missing or
different. The extra engine-only Q/K norm hooks are identity module boundaries.
Actual prefill and first-decode sampling logits and scores match bit-for-bit.

`attention_same_input_replay.json` restores captured Q/K/V strides and verifies
Metal dispatch eligibility: stock attention differs in1257/2048 values;
the maintained standalone Metal operation matches all2048. The normalization,
fixed-token NAR and fixed-latent VAE checks independently isolate their corrections.

`validated_source_state.json` binds the final repeated master to unchanged
production sources, locked dependencies and engine commit, unchanged reference
source revisions/hashes, and all16 frozen model/config/tokenizer file hashes.
The focused regression suite passed66 tests after the final production change.
No production code changed during the final repeats and tensor captures.
All output comparisons use exact array bytes or decoded integer PCM, with
matching shape/sample rate; FLAC container metadata is outside the equality claim.
