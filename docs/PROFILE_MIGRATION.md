# Engine profile migration

Current fork branch: `local-fixes`, pinned at `7ba7e0975cb0bba20a5c7f995fde26d7b096cf79`.
The existing profile commit was amended to enforce the official boundary;
no corrective commit was added. The temporary profile branch remains removed.
Historical receipts below retain their original commit IDs.

## Official boundary correction

`official` now follows untouched upstream `bd90e4ccae671d869b3ecaca6d7e893927d29442`.
Extra MPS cache drains are owned by the ComfyUI profile. Official requires
`rng_device="auto"` and rejects semantic carry, known acoustic latents and
chunk/overlap/blend overrides. The ComfyUI profile enables those capabilities.
The generic primitives and empty model-ready hooks remain reusable API features.
Production `--shim` support is removed; active diagnostic shims are rejected.
Profile identity revision 2 records these policies; revision-1 saved pipelines
need explicit reconstruction/re-export rather than silently changing contracts.

Verification:
- Full engine suite: 216 passed, 11 skipped, 30 subtests passed;
  subsequent upstream-boundary suite: 17 passed (one added pipeline rejection test).
- App suite: 67 passed.
- Independent untouched-upstream comparison covers AR logits, sampling, acoustic
  single/multiple chunks, and even/odd decoder blocks on CPU and MPS.
- Fresh source-checkout ComfyUI-profile riff: ABC/prefix arrays, all 1125 semantic
  tokens and acoustic latents exactly match the previously validated reference.
  Native export is 24-bit, 48 kHz stereo. Evidence:
  `out/parity/official_boundary_check/verified.json` and
  `out/parity/official_profile_engine_tests.log`, `official_upstream_tests.log`,
  `official_profile_app_tests.log`.
- This correction did not rerun the full continuation matrix; its operations
  and algorithms are unchanged. The earlier full-matrix evidence is below.

Current export policy: production now uses the engine's native 24-bit, 48 kHz
FLAC writer. The 16-bit ComfyUI PCM comparisons below describe the preserved
historical validation runs. New production files intentionally use a different
bit depth and encoder; those 16-bit file comparisons are not a current export gate.
Numerical generation still uses the same engine profile.

## Historical profile migration validation

Status at original migration: **complete and verified** on the frozen Apple M5 Pro runtime.

Engine revision: `8ea3770c613e293599137591be24c5d17b9b72b6`, version0.1.7,
branch `profiles/comfyui-yue2-mps-v1`, pushed to the existing engine fork.
This was the migration revision; the current pin is recorded above. Preexisting
uncommitted app work remains in place.

## Design and ownership

The engine owns `NumericalProfile`, `OfficialProfile`, and
`ComfyUIYuE2MPSProfile` (`comfyui-yue2-mps-v1`). The engine default is `official`.
A caller can pass a built-in name or an immutable object implementing the protocol.
Operations are supplied during construction and called per model instance; no
engine or Torch functions are replaced globally. The pipeline's profile cannot
be reassigned through its public property. Construct another pipeline to change it.

The shared sampler accepts explicit carried repetition history. Official behavior
starts that history empty; ComfyUI compatibility includes the carried tokens and
uses request-local CPU RNG. Cancellation, progress, caches and budgets stay in the
shared generation loop. Custom profiles use eager execution so accelerated paths
cannot silently bypass their operations.

Audiogen selects the compatibility profile on MPS by default and calls ordinary
engine continuation methods. Explicit `pipeline.profile` or CLI `--profile`
overrides the choice. Historical text formatting remains in Audiogen; current export uses the engine's
native writer. Its old numerical modules are thin compatibility imports, not duplicated
implementations. Production rendering uses engine profiles and rejects experimental shims.

Take receipts include the profile contract and operation identities, installed
engine revision and recursive source hashes. Pipeline exports retain profile
identity and reject mismatched restoration. The compatibility profile validates
the supported hardware/runtime instead of silently changing operations.

## Measured validation

All paths below are relative to `out/parity/`.

| Check | Evidence | Result |
|---|---|---|
| Engine regressions, including profile isolation and export/reload | `profiles_engine_tests.log` |200 passed,11 hardware/optional skips,30 subtests passed |
| Audiogen regressions | `profile_audiogen_tests.log` |66 passed |
| Initial source-checkout full chain | `profiles_v1_verified/summary.json` and per-stage reports |Four exact prefixes/token arrays/latent arrays/PCM outputs |
| Fresh Git-pinned package full chain | `profiles_verified/chain_verified/summary.json` and per-stage reports |All four stages exact, using own predecessors |
| Additional guitar seeds778/780 and synth fixture777 | `profiles_verified/matrix_summary.json`, individual comparison/prefix reports |Each passed twice |
| Direct repeat comparisons | `profiles_repeatability.json` |All chain stages and matrix cases have identical requests, prefixes, tokens, latents and PCM |
| Original sampled AR with capture enabled | `profiles_sampled_capture/result.json` |1125/1125 original sampled tokens exact |
| Captured model operations and actual sampling scores | `profiles_trace_comparison.json` |Equal weights,1534 exact boundaries, two exact selected-last-row norm equivalents, no missing reference boundaries |
| Same-input attention with captured strides | `profiles_attention_replay.json` |Standalone operation exact2048/2048; stock differs1257/2048 |
| Fresh environment, source and reference preservation | `profiles_validated_source_state.json`, `profiles_verified/environment.json` |Pinned installed engine and app hashes match final take; no ComfyUI import path; reference source and all16 model/config/tokenizer hashes unchanged |
| Distributed implementation and notices | `profile_package_files.json`, `profile_import_audit.json` |Kernel, reference manifest and licenses present; no Comfy imports |

The initial chain used the source checkout while integration was finalized; the
second chain and all matrix/tensor checks used the packaged Git pin without a
local engine source override. Container metadata and timings can differ; decoded
PCM and numerical arrays are compared exactly. No running migration jobs remain.

## Reproduce

```bash
./setup.sh --venv out/parity/profiles-venv --skip-models
out/parity/profiles-venv/bin/python diagnostics/validate_profiles.py --out <new-directory>
```

The validator reads preserved reference outputs under `validation_v1` and uses the
separate reference interpreter only for independent prefix checks. Production
runs solely in the custom environment. Do not overlap MPS inference jobs.

See the engine's `docs/NUMERICAL_PROFILES.md` for API details and runtime scope.
