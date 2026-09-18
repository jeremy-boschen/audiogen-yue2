# Parity diagnostics

The audio comparisons in these historical validators target 16-bit ComfyUI
exports. Current production uses the engine native 24-bit writer, so those
PCM equality gates are not applicable to new exports. Tokens, latent arrays and
prefix comparisons remain useful; see `docs/PROFILE_MIGRATION.md` for the
current verification scope.

Run from the repository root. Model runs execute sequentially on MPS. Each run
requires a new output directory and records arguments, packages, runtime
function bindings, effective-weight hashes and token comparisons. Captures keep
metadata for all layers and full tensors for layer zero at prefill and the first
decode call. `compare_trace.py` compares the saved data on CPU.

Reference imports exist only in `trace_ar.py:load_reference`, run with the
ComfyUI interpreter. `--stack engine` does not enter it. Production code and
vendored operations do not import reference packages. Use `--reference-boot full`
for a reference capture with the actual startup path and custom-node loading.

```bash
.venv/bin/python diagnostics/freeze_reference.py --out out/parity/frozen
.venv/bin/python diagnostics/build_runtime_prefix.py \
  out/parity/frozen/request.json models/YuE2-3B/qwen.tiktoken \
  out/parity/prefix_engine_normalized.npy

~/dev/ai/ComfyUI/.venv/bin/python diagnostics/trace_ar.py \
  --stack reference --reference-boot full --capture \
  --prefix out/parity/prefix_engine_normalized.npy \
  --target out/parity/frozen/greedy45.json --out out/parity/reference_new

.venv/bin/python diagnostics/trace_ar.py \
  --stack engine --shims norm_c --capture --teacher \
  --prefix out/parity/prefix_engine_normalized.npy \
  --target out/parity/frozen/greedy45.json --out out/parity/engine_new

.venv/bin/python diagnostics/compare_trace.py \
  out/parity/reference_new out/parity/engine_new --out out/parity/comparison.json
```

Omit `--teacher` for free-running sampling. Temperature defaults to zero; set
`--temperature 1` and provide the corresponding original target to check the
actual album sampler. Hooks must be validated against an uncaptured reference;
matching initial tokens alone does not validate the complete run.

The original frozen prefix is retained even though it contains the discovered
score-whitespace mismatch. Use the normalized prefix for numerical isolation.
The original 27/1125 result remains reproducible but combined input and numeric
differences. Do not use it as evidence of a purely numerical residue.

## Continuation and independent validation cases

The four-stage song is `songs/burn_it_down_parity`. Its reference workflow
snapshots and source hashes are stored beside the input files. Render with the
same production CLI/flags as the baseline; each stage consumes the prior custom
take. `verify_reference_prefix.py` runs in the reference interpreter and compares
native prefixes against independently extracted workflow inputs.

`prepare_validation.py` extracts guitar seeds778/780 and the historical synth
branch into `songs/parity_validation`. It only reads reference metadata. The
prepared fixtures are already present; use a different `--out` to extract anew.
Once other inference jobs have finished, run the fresh-environment matrix:

```bash
out/parity/fresh-venv/bin/python diagnostics/run_validation.py \
  songs/parity_validation/* --out out/parity/validation_v1
```

This queues each reference request with unique output names, waits for that
specific job, runs two complete custom CLI renders sequentially, and compares
all tokens, latent bytes and decoded PCM. Requests, job IDs, histories, commands,
logs and comparisons are retained. A polling timeout never resubmits a job.

`check_chain.py <render-directory> --out <new-check-directory>` verifies all four
historical stages, including actual reference Plan prefixes and generated carry
links. `freeze_validated_state.py --take <validated-take> --out <state.json>`
checks current production sources against the take's recorded hashes, exact lock
versions and engine commit, absence of ComfyUI from the custom interpreter, and
unchanged reference source/revisions before recording the final source state.

## Engine profile migration

`validate_profiles.py --out <new-directory>` renders the complete chain and two
runs of each additional seed/fixture with the current engine profile. It compares
against the preserved `validation_v1` reference outputs and invokes the actual
reference prefix builder. Run with the fresh pinned interpreter and no other MPS
inference job active. It records installed engine provenance and rejects a ComfyUI
import path in the custom environment.

`trace_ar.py --stack engine --profile` now selects the engine's
`comfyui-yue2-mps-v1` profile. Its diagnostic attention hooks wrap per-instance
operations only; production requires no patch installation. `replay_attention.py`
uses the same profile operation directly on captured Q/K/V with original strides.
