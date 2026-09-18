# Exact-output parity evidence ledger

Current status: **all acceptance gates pass** on the pinned Apple M5 Pro runtime.
See [the completed requirement audit](PARITY_ACCEPTANCE.md). Historical entries
below retain failed experiments and subsequently corrected evidence claims.

Goal remains open: AR, NAR, decoded PCM, continuation chain, additional fixtures,
repeatability and fresh-environment reproduction all require direct evidence.

## 2026-09-18: establish a trustworthy reference

Artifacts live under `out/parity/` (ignored, large tensor dumps). Diagnostic
commands are implemented in `diagnostics/`. Reference-only imports run in the
reference interpreter; the engine path does not import any ComfyUI packages.

- `frozen/manifest.json`: original source revisions, dirty diffs, Python source
  hashes, checkpoint/config/tokenizer hashes, input snapshots and live argv.
- Original discrepancy reproduced with capture: `engine_normc_capture/result.json`,
  27/1125 teacher-forced mismatches, first at 126.
- Fresh live server job `421b687e-67e7-409b-99a7-28c285c54a44` reproduces
  `greedy45.json` exactly. Saved under ComfyUI `output/yue2_takes/` as
  `parity_goal_20260918_greedy.{json,npy}`; prompt/submission/history retained here.
- Both manual and full-startup reference probes initially diverged at token 22.
  Captured and uncaptured manual probes agreed with each other, not the target.
  These initial traces are not reference-parity proof.

## Confirmed input difference: trailing score newline

ComfyUI `runtime.make_plan` uses `abc.strip() or None` before tokenization. Our
request retained the newline. Both prefixes have length 2149 but index 2146 is
7360 in the old engine prefix and 91 in the real runtime prefix. Rebuilding
with the pack's protocol alone previously missed the Plan-node normalization.

`reference_runtime_prefix/result.json` proves 1125/1125 free-running greedy
matches with capture enabled after correcting only that input. It agrees with
both the saved target and the fresh, uncaptured live server execution.

`src/audiogen/render.py:request_for` now normalizes the supplied score at the
same boundary; four regression cases cover outer whitespace, interior
whitespace, empty and absent scores. The engine tokenizer produces the same
normalized prefix as the reference tokenizer (byte-identical saved arrays).

**Correction to the earlier ledger:** all earlier teacher-forcing comparisons
using the old prefix did not have identical context. Their measurements remain
historical observations, but cannot isolate numerical residue alone.

## Numerical lead under measurement

The live startup also patches `F.rms_norm` with AppleSilicon-FP8's standalone
Metal reduction kernel. Thus `norm_c` calls a different operation despite
matching the function name in source. Preliminary same-input traces locate
34 unequal values out of 4,401,152 at layer-zero post-attention RMSNorm, after
identical weights, embeddings, projected/rotated QKV and attention output.
Rechecking with the correct prefix is in progress.

`norm_metal` is an experimental rung using a local MIT-licensed copy of that
kernel, with no ComfyUI imports. It is not yet a production parity claim.

## Confirmed numerical corrections

With the runtime-correct prefix:

| Engine configuration | Teacher-forced mismatches | First |
|---|---:|---:|
| `norm_c` | 16/1125 | 84 |
| `norm_metal` | 17/1125 | 84 |
| `norm_metal,mtl_b` | **0/1125** | none |

`norm_replay.json` proves the vendored normalization kernel returns precisely
all 4,401,152 reference elements on the captured input and weight. Stock
RMSNorm differs in 34. A compact captured-row regression fixture exercises this
specific distinction, including asserting that the Metal kernel actually ran.

After normalization is matched, every prefill layer matches; on first decode,
Q/K/V match and the first differing operation is attention (1257/2048 output
values). Adding `mtl_b` makes all captured comparable values exact, including
raw logits. The only shape difference is final prefill normalization over one
row versus the full sequence; the selected last row is bit-identical.

`engine_greedy_free/result.json`: **1125/1125 free-running greedy tokens exact**,
with capture disabled. Thus this result is not limited to teacher forcing.

These results overturn the earlier claim that Metal attention was ruled out.
Normalization alone slightly worsens the mismatch count yet is necessary for
matching tensors. Argmax counts were not a reliable operation-equivalence test.

The dependency input and hash lock now remove `comfy-kitchen` and add the exact
standalone `mtlflashattn==0.2.0`. Fresh-environment execution remains unverified.

## AR, NAR and VAE stage proof

- `engine_sampled_free/result.json`: original temperature-1 sampler reproduces
  historical `riff45.json`, **1125/1125 tokens**, without teacher forcing.
- `nar_reference/result.json`: captured reference reproduces all live-server
  greedy latents exactly.
- `nar_engine_stock/result.json`: with correct prefix and Metal normalization,
  stock NAR attention differs in 56,028/72,000 final latent values.
- `nar_stock_comparison.json`: first divergent operation is layer-zero NAR
  attention in the first velocity evaluation. Q/K/V and preceding operations
  match. 1,443,466/2,308,096 attention output values differ.
- `nar_engine_mtl/result.json`: changing only NAR SDPA dispatch produces
  **all 72,000 latent values exactly**; captured boundaries match as well.
  This run used the new `fresh-venv`, with no ComfyUI packages installed.
- `decode_engine/result.json`: original engine decoder is 64 samples short.
  Reference Oobleck uses `output_padding=stride % 2`; engine omitted it.
- `decode_engine_padding/result.json`: changing only that padding produces
  **all 4,320,000 float32 waveform values exactly**, 2,160,000 stereo frames.
- `pcm_encoding/result.json`: numpy ties-to-even quantization differs from the
  actual album PyAV/libav encoder in 534 samples. `audio.py` instead calls the
  same standalone encoder with `av==18.1.0` pinned.
- `pcm_encoding/production_result.json`: production serializer produces exact
  decoded PCM for all 2,160,000 stereo frames at 48 kHz.

The maintained `album-mps-v1` profile in `audiogen/parity.py` now configures the
proven RMSNorm, AR attention, NAR attention and decoder-padding operations.
`build_pipeline` selects it by default on MPS when no experimental shims are
active, and records it in native result provenance. Default album FLAC output
uses the actual pinned PyAV encoder, not a hand-reconstructed conversion.
Full CLI validation is running; continuation and additional-fixture proof are
still outstanding. This ledger does not claim goal completion.

## Full baseline acceptance reached; goal still open

The default production CLI (no experimental shims), in the fresh environment:

```bash
out/parity/fresh-venv/bin/python bin/render.py songs/baseline01_riff \
  --out out/parity/e2e --quiet --use-pytorch-cross-attention \
  --disable-smart-memory --reserve-vram 8
```

`album_riff_verified.json`, produced by `diagnostics/compare_take.py`, verifies
identical semantic tokens, bit-identical latents and exact decoded PCM against
the original 09-14 riff. 2,160,000 stereo frames, 48 kHz, decoded s32 PCM SHA-256:
`6824afa6ae1235836d53a0e25bc860bc0bbb6de4f7ed790c0579b30abbb36df2`.

The `run04_today` target used a later ComfyUI core commit with 24-bit FLAC
output, unlike the restored runtime and album's 16-bit output. Reading that
commit established the exact change: `out_stream.format = "s32"`.
The standalone writer supports both via `--pcm-bits 16|24` or manifest
`pcm_bits`. `run04_riff_verified.json` proves exact tokens, latents and 24-bit
PCM using the regenerated take's latents and the same standalone decoder:
`bb9bc9e1ac02e4d5ef50ed841d891908a55a4e8eabc4e1a6c4ab5152849382f1`.
The initial comparison against the wrong bit-depth target is retained as
`e2e_comparison.json`; it was not a synthesis difference.

The original `.venv` now also has `av==18.1.0`, `mtlflashattn==0.2.0`, and no
`comfy-kitchen`. The ComfyUI environment and source were left unchanged.
63 tests pass in the fresh environment; `git diff --check` passes.

### Remaining goal requirements (not yet achieved)

1. Full four-step lineage from regenerated predecessor artifacts. Implement
   per-step score/lyrics inputs and match carried repetition-penalty history
   before measuring the chain. Engine `generate_semantic` currently appends
   carried IDs to the prefix but does not seed the sampling penalty window.
2. Additional fixed seeds and a second input fixture, including full PCM.
3. Repeated complete CLI runs (current repeated token/stage evidence is narrower).
4. Re-run these broader fixtures in the fresh package environment and retain
   all final source/config/weight identities and native receipt verification.
5. Final requirement-by-requirement audit. Do not infer general parity from the
   now-exact 45s riff.

Useful next sources: ComfyUI `run_chain.py` lists original chain FLACs and
LoadTokens boundaries. Extract the embedded workflows, never retype the scores
or lyrics. Its file has a guarded `main()` and reusable FLAC-tag helpers.
No GPU job is intentionally left running at this checkpoint.

## Continuation request fixtures extracted

Added optional `score_file`, `lyrics_file`, and `style_file` step inputs, resolved
relative to the song directory and validated before model loading. Default inputs
remain the song-level files. The request boundary normalizes these inputs in the
same way as the reference plan. A regression exercises changed continuation
inputs, unchanged predecessor inputs, and rejection of a missing score file.

`songs/burn_it_down_parity/song.json` now expresses all four stages. Continuation
inputs were extracted directly from the three historical FLAC prompt tags using
`run_chain.py`'s read-only parser. `reference_workflows.json` retains the source
paths, full source artifact hashes and embedded graphs. Sampling parameters were
checked against the baseline manifest during extraction. The generated token
budgets are 1125, 5875, 2500 and 2050; carried lengths are 0, 1125, 4500 and 4950.
Tail and master use 0.5s carry blending. The manifest uses generated predecessors,
not reference intermediate files. These are request fixtures, not evidence of a
successful chain render. Carried repetition history remains the next runtime gap.

## Explicit carried sampling history implemented; chain run in progress

The standalone continuation sampler in `audiogen/vendor/sampling.py` retains the
reference sampler's actual arithmetic, request-local seed reset and separate
carried penalty window. It imports engine cache/protocol types, not ComfyUI.
Source revision/hash and Apache-2.0 attribution are in `vendor/README.md`.
`audiogen/continuation.py` validates plans, converts carried codec values to model
IDs and passes those IDs explicitly as prior history. Ordinary non-continuation
requests retain their verified engine path.

A controlled regression with identical prefix/logits demonstrates the old and
empty-history paths select the same first token while carried history changes
that choice. It also verifies window expiration, exclusion of prior tokens from
new outputs, and seed isolation from global random draws. 66 tests pass in the
fresh environment. This is focused behavioral evidence, not full-chain proof.

The fresh-environment CLI is running all four generated stages under
`out/parity/chain_v1`; log `out/parity/chain_v1.log`. Its repeated riff has already
passed exact token/latent/PCM comparison (`chain_v1_riff.json`). Continuation
results remain unverified until their artifacts are compared. The trace harness
now supports `--profile` and explicit `--prior` for identical-prefix experiments.

### First full continuation now exact

`chain_v1_grown.json` verifies the 201.2s grown stage against the original
`B_guitar_GROWN_full.flac` and saved `chain_B_grown` tokens/latents. All semantic
IDs, latent bytes and 9,657,600 stereo PCM frames match exactly. Decoded s32 SHA:
`59f35871c386a4b8c574ce548b56b7e657bab0bddb453b3e11306cd6914d1d75`.
This stage consumed the regenerated riff from the same fresh-environment CLI,
not a reference intermediate. Tail and master are still running/unverified.

Additional fixtures are prepared under `songs/parity_validation`: guitar seeds
778 and 780, plus the historical synth-conditioning branch with seed777.
`diagnostics/prepare_validation.py` extracts the original metadata and writes
matching custom manifests/reference API requests; it does not submit or overwrite
reference outputs. These cases have request-equivalence checks, not render proof.

### Tail continuation now exact

`chain_v1_tail779.json`: identical semantic tokens, latent bytes and all
9,651,840 stereo PCM frames (201.08s), including the recorded 0.5s carry blend.
PCM SHA: `b8d3c913a996478ac811a0456b7c68bf8d722c1cb285c56ecb170f72596d6850`.
The independently reconstructed reference prefixes also match native grown and
tail prefixes exactly (`chain_prefixes/{grown,tail779}.json`). Master remains
in flight. Native receipts for future renders now include numerical source
hashes, standalone package/libav versions and effective Metal dispatch settings;
this provenance addition does not change operations. 66 tests still pass.

### Master divergence isolated to the final stage

The first chain run finished. Master does **not** match: custom214.64s
(10,302,720 stereo frames) versus historical209.76s (10,068,480 frames).
`chain_v1_master_pcm.json` records the mismatch. Its 2241-token symbolic prefix
is nevertheless exact (`chain_prefixes/master.json`), and the prior stage's
entire semantic/latent/PCM output is exact. This confines the remaining chain
failure to the final continuation, not accumulated earlier mismatches.

A fresh reference master job `35940bc5-bfff-43dc-b9f1-7c9e5de71951` is running,
with unique `parity_goal_chain_master` audio/token destinations. Its request and
submission are retained under `out/parity/chain_master_reference_*`. It consumes
the historical, verified `chain_B_tail779` at198s. The custom master consumed
its own exact tail. `master_probe/` freezes the7191-token AR prefix,4950-token
prior history and custom new-token target for subsequent isolated traces.
No claim of four-stage parity is made.

### Correction: master lyric whitespace, not numerical divergence

The live reference master reproduced the historical209.76s PCM exactly
(`master_reference_verified.json`). Both the isolated engine and full-startup
reference probe, when fed our frozen prefix, instead reproduced the same416 new
tokens; first disagreement with the live result at new-token32.
`master_trace_comparison.json` shows equal effective weights and no differences
at the captured comparable tensor boundaries. Thus the probe input was wrong.

Actual `runtime.make_plan` strips **ABC only**, retaining style and lyrics as
provided. Master's embedded lyrics end in a newline. Our new per-step loader
incorrectly stripped it. The initial prefix verifier repeated that assumption,
so `chain_prefixes/master.json` is INVALID evidence of reference input equality.
The corrected verifier (`master_corrected_verifier.json`) establishes a token
change at symbolic-prefix index289:198 versus271, despite equal2241-token length.
This is the reason not to equate prefix lengths or duplicated normalization code
with independently validated requests.

Explicit `style_file`/`lyrics_file` inputs now preserve raw content; score trimming
remains unchanged. Legacy song-level defaults retain their existing normalized
file convention. All extracted parity fixtures select explicit text files so
reference request strings are represented exactly. The riff's fixture text was
re-extracted from its original graph rather than carrying added file newlines.
The regression now requires the supplied lyric newline to survive.66 tests pass.
`master_probe/corrected_prefix_verified.json` establishes true prefix equality;
corrected-input free-running AR verification is running next.

`master_engine_corrected_input/result.json` now proves all294 new master tokens
match the live reference exactly, free-running at temperature1 with seed779.
Only the lyric whitespace/prefix changed relative to the failing isolated run.
The full CLI is rerunning all four stages from scratch under `out/parity/chain_v2`
(log `chain_v2.log`), using the fresh environment and generated predecessors.
At launch its process was PID21314, shell session53207; inspect/poll that live job
rather than starting a duplicate. Full corrected-master latent/PCM proof remains
pending. The wheel build also contains both standalone implementations and their
license/attribution files (`out/parity/wheel`, `wheel_build.log`).

After the chain completes, compare master against both the historical FLAC and
new `parity_goal_chain_master.{json,npy}` reference intermediates. Then run the
prepared additional-case/repeat matrix with `diagnostics/run_validation.py`.
Do not run GPU jobs concurrently with the active chain.

The prefix verifier now calls the actual reference `runtime.make_plan` with its
real tokenizer, using supplied ABC so no model weights load. It no longer copies
normalization logic. `master_probe/actual_runtime_prefix_verified.json` confirms
the corrected input through that entrypoint. The new full chain's riff again
passes exact tokens/latents/PCM (`chain_v2_riff.json`).

## Full four-stage lineage verified

`chain_v2_verified/summary.json` and its four per-stage reports establish exact
semantic tokens, latent bytes and decoded PCM for the complete chain, rendered
from scratch in the fresh environment with generated predecessors. Each native
prefix also matches the actual reference `runtime.make_plan` execution on the
independent embedded workflow. The final master is209.76s,10,068,480 stereo
frames at48kHz, decoded s32 SHA:
`4e53462460fc12714d270affe14619b102ddcc158cc8bde33b2b23931fdbe930`.

The listening comparison at `http://127.0.0.1:8766/` now includes all four stages
and original counterparts. Copies were SHA-256 checked against source files.

The additional-fixture matrix is running under `out/parity/validation_v1`, driven
by the still-active shell session47917. Log: `out/parity/acceptance_queue.log`.
Its first reference job is `c019926a-c01e-49d7-a182-70c568f73bdd` (guitar seed778).
It will compare two full custom runs per case, including prefixes via the actual
reference Plan function. Do not submit duplicate GPU jobs while it is active.
Remaining: this matrix, final repeatability/source-provenance audit and any gaps
identified by the requirement-by-requirement completion audit. Goal remains open.

### Acceptance runner state

Both seed778 custom runs now pass every exact comparison and the actual reference
prefix check (`validation_v1/guitar_seed778/`). Seed780 reference job
`0884c889-d801-4f8a-9627-3c56b40d5b5a` is next/in progress.
The acceptance coordinator is PID21573, shell session47917.

A final full corrected-chain repeat is queued behind that coordinator, shell
session14619, log `chain_repeat_queue.log`. It waits for PID21573 to exit and
requires all six matrix comparisons to have passed before rendering `chain_v3`
and running `check_chain.py` into `chain_v3_verified`. Do not launch either job
again while these handles/processes are live. This gives repeated complete
lineage evidence, including the corrected master, rather than inferring that
property from repeated short riffs alone.

`docs/PARITY_ACCEPTANCE.md` maps all explicit requirements to current evidence
and names remaining gates. It is an audit checklist, not a completion claim.

### Final audit identified a missing direct score comparison

Earlier reference AR traces captured model logits but did not capture sampler
scores; only the engine teacher-forced trace had `scores` entries. Thus the
acceptance table's broad score-comparison claim needs additional direct evidence.
The trace harness now records the actual distribution call on both free-running
paths, plus effective model-operation bindings. This instrumentation does not
replace or reconstruct the sampler. After the active full-chain repeat finishes,
run paired captured baseline temperature1 probes against original `riff45.json`
and verify both complete token sequences and their score tensors. This also
checks capture invariance under the original sampled configuration.

## Additional fixture and repeated-run matrix passed

`validation_v1/summary.json` contains six successful complete comparisons:
guitar seeds778/780 and historical synth-style seed777, each rendered twice in
the fresh environment. All prefixes, semantic tokens, latent bytes and decoded
PCM are exact against their live reference cases. `repeatability_matrix_verified.json`
also directly compares each pair's arrays, requests and numerical identities.
`synth_historical_verified.json` independently matches the original synth FLAC,
PCM SHA `0dd350fdf61095136227b071c73da0a5aa715903bf790240bd34a27e83fc5d3a`.

Source audit confirms current production Python hashes match the validated chain,
all standalone dependencies match the lock, the installed engine matches the git
pin, no ComfyUI package/import path is present, and reference source/revisions
remain unchanged. `standalone_attention_source_verified.json` additionally
matches all five installed attention source files between environments.

Active remaining jobs: full-chain repeat coordinator PID23006/session14619,
render child PID23332; log `chain_v3.log`. Paired original-sampler score captures
are queued behind that coordinator in shell session10771, log `final_score_probe_queue.log`. They will not run unless
the complete repeated chain passes. Final source freeze will also rehash model,
config and tokenizer files against the original frozen hashes.

### Corrected full-chain repeat verified

The existing chain_v3 coordinator exited successfully. All four actual-reference
prefix checks and token/latent/PCM comparisons pass in `chain_v3_verified`.
Direct custom-v2 versus custom-v3 comparisons also pass for every stage, including
request equality, array dtype/shape/bytes and decoded integer PCM with sample rate:
`out/parity/repeatability_chain_verified.json`. The original-sampling reference
capture is now running under the existing final-score coordinator; no job was
restarted. Direct sampler score comparison remains pending.

### Final sampled capture and completion audit

Both original-temperature sampled captures reproduce1125/1125 target tokens.
Effective weights match. All1536 reference trace boundaries are accounted for:
1534 exact and two final-norm selected-last-row equivalents. Prefill and first
new-token sampling logits and actual distribution scores are bit-identical.
Evidence: `final_sampled_trace_comparison.json` and both capture result files.
The sequential coordinator exited successfully; no missing-left boundary was
accepted as success during the final inspection.

The same-input attention replay restored original captured strides, confirmed
Metal eligibility, and measured1257/2048 stock mismatches versus zero for the
standalone operation (`attention_same_input_replay.json`). Final source audit
against chain_v3 master passes, including all16 model/config/tokenizer files,
reference sources, exact package pins, engine revision and no Comfy import path.
The complete chain and three additional cases have direct repeatability proof.
No known output discrepancy remains within this explicitly tested runtime/scope.

### Engine profile migration completed

Fork revision `8ea3770c613e293599137591be24c5d17b9b72b6` introduces instance-owned
profiles and the shared sampler's explicit carried history. Audiogen is pinned to
it; its maintained MPS path no longer patches engine operations. Both the initial
source chain and fresh installed chain match all four reference stages. Seeds778,
780 and synth777 match twice each in the fresh environment. Direct repeats,
original sampled token/logit/score traces and captured-stride attention replay pass.
Engine200 tests and Audiogen66 tests pass; optional/hardware skips remain recorded.
Final source/environment/reference preservation audit passes. See
`PROFILE_MIGRATION.md` for the complete evidence table and canonical commands.
