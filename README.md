# audiogen-yue2

Create music with YuE2 using song manifests, reusable scores, continuation chains
and optional LoRA adapters. Each take includes 24-bit, 48 kHz stereo FLAC, the
score, semantic tokens, acoustic latents and a record of its inputs and settings.

## Install

```sh
./setup.sh --skip-models    # use existing models/ links
source .venv/bin/activate
```

For a new installation, run `./setup.sh` to use the shared model location in
`env/pins.env`, or `./setup.sh --models /path/to/YuE2` to link existing weights.
The weights directory must contain `YuE2-3B` and `YuE2-Vae`. Setup recreates the
selected environment from the pinned lockfile; it is not an environment updater.
The engine revision and runtime are pinned in `env/`.

## Make a song

Copy `songs/example` to a new song directory and edit:

- `style.txt`: instruments, mood, vocal style and production direction.
- `lyrics.txt`: lyrics and section labels.
- `song.json`: seed, take lengths, steps and engine settings.
- `score.abc` (optional): a supplied score; omit it to let YuE2 plan one.

```sh
.venv/bin/python bin/render.py example --dry-run
.venv/bin/python bin/render.py example --out ../audiogen/takes
.venv/bin/python bin/render.py example --label 'Evening take' --out ../audiogen/takes/evening
```

Use a new output directory for each take you want to keep. Rendering to the same
song/step path can overwrite an earlier take. Output defaults to this repo's
ignored `out/` directory if `--out` is omitted.

`seconds` is a generation budget, not a guaranteed duration. The model can stop
early. `max_tokens` specifies the new-token budget directly; there are 25 semantic
tokens per second. If both forms are supplied, their budgets must agree.

## Continue a take

Define steps in dependency order:

```json
{
  "id": "my_song",
  "seed": 777,
  "cot": "full",
  "steps": [
    {"id": "riff", "seconds": 45},
    {"id": "grown", "seconds": 180, "carry_from": "riff", "carry_seconds": 30}
  ]
}
```

`carry_seconds` keeps that much of the previous take from its start. The new-token
budget is the target length minus the carried length. `carry_tokens` provides
exact token control. The source take must actually contain enough tokens.

```sh
.venv/bin/python bin/render.py my_song --step grown --out ../audiogen/takes/new-session
```

This renders the selected step and its predecessors. It does not resume an
existing output directory. A step can choose `score_file`, `style_file`, or
`lyrics_file` relative to the song directory. Explicit lyric/style files retain
whitespace; scores have outer whitespace trimmed. `blend_seconds`,
`chunk_seconds`, and `overlap_seconds` control acoustic continuation.

## Profiles and resources

`pipeline.profile` in the manifest or `--profile` selects engine behavior:

- `comfyui-yue2-mps-v1`: default on MPS; supports continuation. Requires the
  pinned Apple M5 Pro / macOS 26.6.2 runtime and unquantized torch execution.
- `official`: upstream engine behavior; requires automatic RNG selection and
  does not accept continuation or custom chunk/blend overrides.

The engine rejects unsupported profile/runtime combinations. A seed alone does
not reproduce a take across different profiles, models, hardware or runtimes.
Keep the saved inputs, artifacts and settings for takes you value.

### What this means on Windows

`comfyui-yue2-mps-v1` is rejected at validation off Apple Silicon, so a
Windows/CUDA box always runs `official`, and that is not a setting anyone can
change. Two things follow, and a song written on the Mac will hit both:

- **No growing a take out of an earlier one.** Any step with `carry_from` is
  refused, before any weight is loaded, naming the steps to remove. This one is
  a real capability and there is no way around it on this hardware.
- **`rng_device` is forced to `auto`.** A song that pins it to `cpu` still
  renders; the setting is dropped and the change is printed. It only decides
  which device draws the sampling noise, and the seeds do not reproduce across
  the two machines regardless.

`setup.sh` picks the lockfile from `uname`: `env/requirements.lock.txt` on
macOS, `env/requirements-win-cuda.lock.txt` on Windows. They differ by exactly
two packages -- the Windows one drops the Metal attention kernel and takes
torch from download.pytorch.org, because the torch on PyPI is CPU-only on
Windows and would install cleanly, import cleanly, and never touch the GPU.

The same seed does not give the same take on the two machines. It was never
going to: different profile, different kernels, different hardware.

Use `--device`, `--backend`, `--memory-budget-gib`, and `--offload-ar` /
`--no-offload-ar` for explicit resource choices. The same engine options can be
stored under `pipeline` in `song.json`. CLI choices override manifest values.
Sampling controls live under `abc_sampling` and `semantic_sampling`;
engine generation controls live under `generation_config`.

## Adapters

```sh
.venv/bin/python bin/render.py example --lora /path/to/adapter.safetensors:nar:0.8
.venv/bin/python bin/render.py example --no-lora
```

An adapter declares its `ar` or `nar` branch and optional strength. `--lora` can
be repeated and adds to the manifest's `lora` list. `--no-lora` removes manifest
adapters. Missing files and incompatible branch declarations are rejected.

## Python API

```python
from pathlib import Path
from audiogen.song import load
from audiogen.render import build_pipeline, render_step

song = load(Path("songs/example"))
pipe = build_pipeline(Path("models"), song)
try:
    previous = None
    for step in song.steps:
        take = render_step(pipe, song, step, previous)
        take.write(Path("out/session") / song.id / step.id)
        previous = take
finally:
    pipe.close()
```

The CLI additionally writes a song-level `provenance.json`. Per-take artifacts
record effective engine configuration, model identities, inputs and hashes.

## Development

```sh
.venv/bin/python -m pytest -q
```

`./setup.sh --dev` installs the sibling `../YuE` checkout editable. Use the
pinned installation for saved production takes.

## Credits

Made by [Jeremy Boschen](https://github.com/jeremy-boschen), built with Claude, which wrote most of the code.

YuE2 is made by the [Multimodal Art Projection (m-a-p)](https://huggingface.co/m-a-p) team:
[code](https://github.com/multimodal-art-projection/YuE),
[model](https://huggingface.co/m-a-p/YuE2-3B), [VAE](https://huggingface.co/m-a-p/YuE2-Vae) and
[technical report](https://github.com/multimodal-art-projection/YuE/blob/main/docs/technical_report.pdf).
This project is independent of them. It runs YuE2 through a
[fork of their inference library](https://github.com/jeremy-boschen/YuE), pinned in `env/`.

The microscope explorer also uses [three.js](https://threejs.org) (MIT, vendored in
`src/audiogen/explore_assets/vendor/three/` with its license),
[Demucs](https://github.com/facebookresearch/demucs) to estimate when parts come in,
[SheetSage](https://github.com/chrisdonahue/sheetsage) for chords, and
[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) with the
[Qwen3 forced aligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), run through
[audio.cpp](https://github.com/0xShug0/audio.cpp), for recognized words. Each belongs to its authors
under its own license.
