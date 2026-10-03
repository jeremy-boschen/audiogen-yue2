"""A stack fingerprint: will a seed give the same song on this machine as it did then?

The same seed only reproduces a take where the maths underneath is the same: the
chip, the OS and its GPU drivers, torch, the Metal kernels, the engine's code and
the weights. A list of versions can only say any of those *might* matter; this
runs them. A fixed probe goes through the real model and the profile's own
operations, and its outputs are hashed. Equal hashes mean the stack computes the
same bytes; a different hash means a seed may now give a different song.

The probe has two parts:

* kernels: each profile operation (RMSNorm, AR and NAR attention) on fixed seeded
  tensors at the model's own sizes, including lengths past the 1024 and 4096
  thresholds where the MPS profile switches to Metal flash attention. A short
  render never reaches those, and a canary that stayed under a threshold once
  passed under the wrong launch flags (git d9c66f8).
* model: a tiny render through every stage (score, semantic, acoustic, decode)
  with a fixed request, seed and sampling, so the real weights, sampler, RNG
  and decoder are in it too.

Measuring loads the model, so a result is kept per `key`: everything that could
change the maths, read without importing torch or the engine. A key that has
been measured is reused; a new key (an OS update, a new torch, an edited engine)
is measured again the next time something renders.

    fingerprint.current(models, options, cache)   # the record, measured if needed
    fingerprint.lookup(models, options, cache)    # the record, or None, never measures
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path


# Raise this when the probe itself changes: its hashes are then not comparable
# with older ones, which is different from the stack having changed.
PROBE = 1
SEED = 20261003
STYLE = "Indie pop, warm synths, 108 BPM"
LYRICS = "[Verse]\nPorch light on the landing\nKeys in the paper cup\n"
ABC_SAMPLING = {"min_tokens": 0, "max_tokens": 48}
SEMANTIC_SAMPLING = {"min_tokens": 0, "max_tokens": 50}
# Key/value lengths for the attention probes: under 1024, between the two
# thresholds, and past 4096 (comfyui-yue2-mps-v1 metal_eligible).
LENGTHS = (900, 1500, 4200)
# Settings that change what a render computes; device placement only is left out.
OPTIONS = ("profile", "backend", "device", "quantization", "vae_core_frames")
ENV_PREFIXES = ("PYTORCH", "TORCH", "MPS", "MTLFLASH", "OMP", "MKL", "CUDA")
WIDTH = 16          # as audiogen.hashes; kept here so a lookup needs no numpy


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:WIDTH]


def _run(*command) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)      # finds a package without running it
    return Path(spec.origin).parent if spec and spec.origin else None


def _torch_git() -> str | None:
    folder = _package_dir("torch")
    text = (folder / "version.py").read_text() if folder and (folder / "version.py").is_file() else ""
    line = next((l for l in text.splitlines() if l.startswith("git_version")), "")
    return line.split("=", 1)[1].strip().strip("'\"") or None if line else None


def _source_digest(name: str) -> str | None:
    """The engine's own source, as installed: an editable checkout can change under one version."""
    folder = _package_dir(name)
    if folder is None:
        return None
    sha = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".json", ".metal") and "__pycache__" not in path.parts:
            sha.update(str(path.relative_to(folder)).encode() + b"\0" + path.read_bytes() + b"\0")
    return sha.hexdigest()[:WIDTH]


def stack() -> dict:
    """What this machine runs, in words: shown beside a fingerprint, never compared."""
    mac = platform.mac_ver()[0]
    return {
        "machine": socket.gethostname().removesuffix(".local"),
        "os": f"macOS {mac}" if mac else platform.platform(terse=True),
        "os_build": _run("sysctl", "-n", "kern.osversion") if mac else platform.version(),
        "chip": (_run("sysctl", "-n", "machdep.cpu.brand_string") if mac else platform.processor()) or platform.machine(),
        "python": platform.python_version(),
        "torch": _version("torch"),
        "torch_git": _torch_git(),
        "mtlflashattn": _version("mtlflashattn"),
        "yue2": _version("yue2-infer"),
    }


def _weights(models: Path) -> dict:
    out = {}
    for path in sorted(Path(models).glob("*/model.safetensors")):
        info = path.stat()
        out[path.parent.name] = [info.st_size, int(info.st_mtime)]
    return out


def key(models: Path, options: dict) -> str:
    """Everything that could change the probe's bytes, without loading anything heavy."""
    facts = {k: v for k, v in stack().items() if k != "machine"}
    payload = {"probe": PROBE, "stack": facts, "engine": _source_digest("yue2"),
               "weights": _weights(models),
               "env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith(ENV_PREFIXES)},
               "options": {k: options.get(k) for k in OPTIONS}}
    return _digest(json.dumps(payload, sort_keys=True).encode())


def _read(cache: Path) -> dict:
    try:
        return json.loads(Path(cache).read_text())
    except (OSError, ValueError):
        return {}


def lookup(models: Path, options: dict, cache: Path) -> dict | None:
    """This machine's fingerprint for these settings if it has been measured; None if not."""
    return _read(cache).get(key(models, options))


def current(models: Path, options: dict, cache: Path) -> dict:
    """This machine's fingerprint for these settings, measuring it the first time."""
    wanted = key(models, options)
    known = _read(cache).get(wanted)
    if known is not None:
        return known
    record = {**measure(models, options), "key": wanted}
    entries = _read(cache)
    entries[wanted] = record
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial")
    partial.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    os.replace(partial, cache)
    return record


def kernels(pipe) -> dict:
    """The profile's operations on fixed tensors shaped as the model shapes them."""
    import torch

    from . import hashes

    profile, device = pipe.profile, pipe.device
    config = json.loads((Path(pipe.model_dir) / "config.json").read_text())
    heads, groups, dim, width = (config["num_attention_heads"], config["num_key_value_heads"],
                                 config["head_dim"], config["hidden_size"])
    generator = torch.Generator("cpu").manual_seed(SEED)

    def tensor(*shape):
        return torch.randn(shape, generator=generator).to(torch.bfloat16).to(device)

    def out(x):
        return hashes.hash_array(x.detach().float().cpu().numpy())

    parts = {}
    with torch.inference_mode():
        parts["rms_norm"] = out(profile.rms_norm(tensor(64, width), tensor(width), 1e-6))
        parts["rms_norm_head"] = out(profile.rms_norm(tensor(64, heads, dim), tensor(dim), 1e-6))
        # AR prefill is causal; each later token is one query against the cache (modeling_yue2).
        parts["ar_prefill"] = out(profile.ar_attention(tensor(1, heads, 512, dim), tensor(1, groups, 512, dim),
                                                       tensor(1, groups, 512, dim), is_causal=True))
        for n in LENGTHS:
            parts[f"ar_step_{n}"] = out(profile.ar_attention(tensor(1, heads, 1, dim), tensor(1, groups, n, dim),
                                                             tensor(1, groups, n, dim)))
        # NAR: 256-row query blocks against every frame, heads already repeated on MPS (nar.attention).
        parts["nar_causal"] = out(profile.nar_attention(tensor(1, heads, 256, dim), tensor(1, heads, 256, dim),
                                                        tensor(1, heads, 256, dim), is_causal=True))
        for n in LENGTHS:
            parts[f"nar_{n}"] = out(profile.nar_attention(tensor(1, heads, 256, dim), tensor(1, heads, n, dim),
                                                          tensor(1, heads, n, dim)))
    return parts


def render(pipe) -> dict:
    """A tiny take through every stage, with a request and sampling that never change."""
    from yue2.protocol import SongRequest

    from . import hashes

    plan = pipe.plan(request=SongRequest(style=STYLE, lyrics=LYRICS, seed=SEED, id="fingerprint"),
                     abc_sampling=ABC_SAMPLING)
    parts = {"abc": hashes.hash_tokens(plan.abc_ids)}
    semantic = pipe.generate_semantic(plan, sampling=SEMANTIC_SAMPLING)
    parts["semantic"] = hashes.hash_tokens(semantic.tokens)
    if semantic.tokens:
        latents = pipe.synthesize(semantic)
        parts["latent"] = hashes.hash_array(latents)
        parts["pcm"] = hashes.hash_array(pipe.decode(latents))
    return parts


def measure(models: Path, options: dict) -> dict:
    """Run the probe on a pipeline of its own: no adapters, default generation settings."""
    from .render import build_pipeline
    from .song import Song

    clock = time.perf_counter()
    settings = {k: options[k] for k in OPTIONS if options.get(k) is not None}
    pipe = build_pipeline(Path(models), Song(root=Path(models), id="fingerprint", seed=SEED, pipeline=settings),
                          progress=False)
    try:
        parts = {**kernels(pipe), **render(pipe)}
        identity = {"profile": pipe.profile.name, "backend": pipe.backend, "device": str(pipe.device),
                    "quantization": pipe.quantization}
    finally:
        pipe.close()
    return {"probe": PROBE, "hash": _digest(json.dumps(parts, sort_keys=True).encode()), "parts": parts,
            "stack": stack(), "settings": identity, "seconds": round(time.perf_counter() - clock, 1),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def same(a: dict | None, b: dict | None) -> bool | None:
    """True or False when two fingerprints can be compared; None when they cannot."""
    if not a or not b or a.get("probe") != b.get("probe"):
        return None
    return a.get("hash") == b.get("hash")


def differs(a: dict, b: dict) -> list[str]:
    """Which parts of the probe disagree: the first is where the maths moved."""
    left, right = a.get("parts", {}), b.get("parts", {})
    return [name for name in left if left.get(name) != right.get(name)]
