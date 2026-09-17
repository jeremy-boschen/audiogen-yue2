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

## What is and is not reproducible

Pinned and verified: the environment, and a take regenerated inside it.

Not portable across machines without measurement: the same seed on a different
device. Every RNG source is pinnable (`rng_device="cpu"`, and the NAR noise is
already CPU), but the logits reaching the sampler come from device-specific
kernels, and in bf16 one flipped logit diverges forever. To move a take between
machines, move the artifact rather than the seed: `semantic.npy` skips the
autoregressive amplifier, `latent.npy` leaves only a deterministic decode.
