# audiogen-yue2

Music generation on the official YuE2 inference library, with no ComfyUI in the
stack. Everything here is pinned so a take can be regenerated exactly.

```sh
./setup.sh              # builds .venv, fetches weights, verifies the fork patches
source .venv/bin/activate
```

## Layout

| path | what it is |
|---|---|
| `env/pins.env` | every pinned revision: python, the fork commit, torch, weight SHAs |
| `env/requirements.lock.txt` | 35 packages, 852 hashes, installed with `--require-hashes` |
| `setup.sh` | builds the environment from those pins; idempotent |
| `src/audiogen/` | our conventions over the library (see the module docstring) |
| `songs/<name>/` | what defines a song: score, lyrics, style, request, manifest |
| `docs/` | how this relates to the ComfyUI stack, and what reproduces |

## The two-repo split

- **[audiogen-comfyui](https://github.com/jeremy-boschen/audiogen-comfyui)** is the
  frozen baseline. It reproduces the album *Forgives at Five* bit-for-bit on
  decoded PCM. It changes only when reproduction of that album needs it.
- **This repo** is where new work happens.

They do not produce the same audio and are not meant to. See
[docs/MATCHING_COMFYUI.md](docs/MATCHING_COMFYUI.md) for the bisected reasons,
including the one that no amount of porting fixes.

## The library fork

`setup.sh` installs [jeremy-boschen/YuE](https://github.com/jeremy-boschen/YuE) at
the commit in `pins.env`. That fork carries five commits on top of upstream, each
doing one thing, each inert by default:

| commit | what it adds |
|---|---|
| stage guard | drains the MPS allocator between stages, which otherwise makes a take unreproducible |
| `carry=` | continue a take from a previous take's semantic tokens |
| chunk pinning | pin the acoustic chunk layout; carry known latents |
| `on_model_ready` | a hook to attach adapters without reaching for a private attribute |
| `rng_device` | force which device's RNG draws tokens, so a seed can travel |

Engine changes go in the fork; our conventions go here. Every fork commit is a
rebase cost forever, so the line is kept deliberately.

`./setup.sh --dev` installs the fork editable from `../YuE` for library work.
Never use it for a take you intend to keep -- the pin is what makes a run
reproducible.

## Rendering a song

A song is a directory; `song.json` is its contract.

```sh
bin/render.py burn_it_down              # every step, in order
bin/render.py burn_it_down --step grown # one step, plus the chain it carries from
bin/render.py burn_it_down --dry-run    # resolve and check the manifest, load no weights
bin/render.py burn_it_down --out ~/dev/projects/audiogen/takes
```

Audio never lands in this repo. The repo holds what *defines* a song -- score,
lyrics, style, manifest; the studio holds what it renders to.

### Steps and carrying

```json
{
  "id": "burn_it_down",
  "seed": 777,
  "cot": "full",
  "generation_config": {"rng_device": "cpu"},
  "semantic_sampling": {"temperature": 1.0, "penalty_window": 50, "min_tokens": 200},
  "steps": [
    {"id": "riff",  "seconds": 45.0},
    {"id": "grown", "seconds": 201.2, "carry_from": "riff",  "carry_seconds": 45.0},
    {"id": "tail",  "seconds": 201.1, "carry_from": "grown", "carry_seconds": 180.0, "seed": 779}
  ]
}
```

`seconds` is the length the take should **end up**. `carry_seconds` is how much of
the earlier take is kept, measured from its start, and is always explicit -- a
sentinel meaning "all of it" reads fine until the source length changes underneath
it. The runner converts to the engine's `max_tokens`, which counts **new** tokens
rather than the total, so a manifest states the thing a person actually means.

### Engine units

Seconds are a convenience, and a convenience that cannot be bypassed is a cage.
Every derived value has an escape hatch naming the engine's own unit, and the
hatch wins outright:

| convenience | engine unit |
|---|---|
| `"seconds": 200.0` | `"max_tokens": 3875` |
| `"carry_seconds": 45.0` | `"carry_tokens": 1125` |

Both spellings of the same chain resolve to the same plan, and a step may give
both -- a manifest is allowed to spell out what it means. But they must agree: a
mismatch is an **error naming both values**, never a silent precedence rule, since
a precedence rule is only ever discovered once the two have already drifted.

```
step 'b' disagrees with itself: seconds=200.0 less 1125 carried is 3875 new
tokens, but max_tokens=5000 was given
```

Unknown manifest keys are rejected too, so a typo cannot silently leave a default
in place.

Everything the engine already names well -- `chunk_seconds`, `overlap_seconds`,
`blend_seconds`, `penalty_window`, `min_tokens`, `rng_device`, the whole
`generation_config` -- passes through under its own name. The wrapper abstracts
gotchas, not the engine.

A carrying step hands the earlier take's semantic tokens to generation *and* its
latents to synthesis, then asserts the carried tokens came back verbatim. If the
continuation silently failed to take, the render stops rather than quietly
producing a fresh song at the right length.

`--dry-run` validates the whole manifest -- carry order, carry length, leftover
budget -- before a single weight loads. A bad manifest fails in milliseconds, not
twelve minutes in.

Each run writes `provenance.json` next to the takes: resolved torch and
`yue2-infer` versions, device, seeds, sampling, hashes of style and lyrics, and
every stage hash per take.

## The canary

`bin/canary.py` renders `songs/_canary` and hashes **every stage boundary**, not
just the audio:

```sh
bin/canary.py            # verify against songs/_canary/expected.json
bin/canary.py --record   # rewrite it, only when a change is intended
```

Per stage, because the acoustic stage attends over the whole sequence: a semantic
token that first differs at 200s still changes latent frame 0. An end-to-end audio
hash can only say "different"; the stage hashes say *where*, and everything
downstream of the first disagreement differs as a consequence rather than
independently.

Run it after any rebase onto upstream, any new fork commit, and any change to the
lockfile. It is also the instrument for a cross-machine comparison -- run it on two
boxes and the first stage that disagrees is the answer.

### Why this length

The fixture is deliberately long enough to cross this stack's length-dependent
branches. These were read out of the library, not assumed:

| branch | threshold | at 25 tok/s | where |
|---|---|---|---|
| EOS suppressed below `min_tokens` | 200 tokens | 8.0s | `sampling.py` |
| repetition `penalty_window` slides | 50 tokens | 2.0s | `sampling.py` |
| NAR attention query-blocks (MPS/CPU; CUDA does the full sequence) | 256 frames | 10.2s | `nar.py` |
| NAR splits into multiple chunks | `(24576 - prefix - 3) // 2` | ~475s | `protocol.py` |

`expected.json` records which of these the recorded take actually crossed, so the
coverage is a measurement rather than a claim. The last one needs an eight-minute
render and is **not** crossed; the same stitching path is reachable cheaply by
forcing `chunk_frames`.

The precedent: the ComfyUI canary was nearly useless at 25s because it sat under a
1024-token kernel threshold and **passed under the wrong launch flags**. That
specific gate belonged to a node pack not used here, but the lesson generalises --
a probe shorter than the behaviour it is meant to detect proves nothing.

Rendered audio is not committed. The stage hashes localise a regression better than
a waveform, and `reference/semantic.npy` re-derives the audio exactly;
`--record` writes the FLAC to `out/`, which is gitignored.

## What is and is not reproducible

Pinned and verified: the environment, and a take regenerated inside it.

Not portable across machines without measurement: the same seed on a different
device. Every RNG source is pinnable (`rng_device="cpu"`, and the NAR noise is
already CPU), but the logits reaching the sampler come from device-specific
kernels, and in bf16 one flipped logit diverges forever. To move a take between
machines, move the artifact rather than the seed: `semantic.npy` skips the
autoregressive amplifier, `latent.npy` leaves only a deterministic decode.
