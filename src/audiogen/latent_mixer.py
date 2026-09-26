"""Edit a finished take's latent, channel by channel, before it is decoded.

The acoustic latent is [frames, 64] at 25 frames per second. Nobody designed
what its 64 channels mean: the VAE learned them. These edits are how to find
out by ear. Every edit is relative to the channel's own statistics over the
whole take, so "+1" means one of that channel's standard deviations, and
"muted" means held at its average (not zero, which is not neutral).
"""
from __future__ import annotations

import numpy as np

CHANNELS = 64


def stats(latent: np.ndarray) -> dict:
    return {"mean": latent.mean(axis=0), "std": latent.std(axis=0)}


def apply(latent: np.ndarray, edits: dict, whole: dict, basis: dict | None = None) -> np.ndarray:
    """Return an edited copy of ``latent``; ``whole`` is ``stats`` of the full take.

    edits: {"offset": [64] in stds, "gain": [64] around the mean,
            "mute": [64] bools, "solo": [64] bools, "master_gain": float}
    Solo wins over mute: if any channel is soloed, every other channel is held
    at its mean. Gain scales a channel's movement around its mean; offset then
    shifts it. Master gain scales every channel's movement around its mean.
    "directions": [K] moves along ``basis``'s principal directions, each in units
    of that direction's own spread across the takes it was computed from.
    """
    mean, std = whole["mean"], whole["std"]
    out = latent.astype(np.float32, copy=True)
    offset = np.asarray(edits.get("offset", np.zeros(CHANNELS)), np.float32)
    gain = np.asarray(edits.get("gain", np.ones(CHANNELS)), np.float32)
    mute = np.asarray(edits.get("mute", np.zeros(CHANNELS, bool)), bool)
    solo = np.asarray(edits.get("solo", np.zeros(CHANNELS, bool)), bool)
    master = float(edits.get("master_gain", 1.0))
    for vector in (offset, gain, mute, solo):
        if vector.shape != (CHANNELS,):
            raise ValueError(f"every per-channel edit needs {CHANNELS} values")
    held = ~solo if solo.any() else mute
    out = mean + (out - mean) * gain * master + offset * std
    out[:, held] = mean[held]
    moves = np.asarray(edits.get("directions") or [], np.float32)
    if moves.size:
        if basis is None or len(moves) > len(basis["components"]):
            raise ValueError("direction edits need a basis with that many directions")
        out = out + (moves * basis["std"][:len(moves)]) @ basis["components"][:len(moves)]
    return out.astype(np.float32)


def neutral() -> dict:
    return {"offset": [0.0] * CHANNELS, "gain": [1.0] * CHANNELS,
            "mute": [False] * CHANNELS, "solo": [False] * CHANNELS, "master_gain": 1.0}


def signal_stats(audio: np.ndarray, rate: int) -> dict:
    """Signal statistics of a decoded excerpt: describe it, never judge it."""
    x = audio.mean(axis=1) if audio.ndim == 2 else audio
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    total = spec.sum() or 1.0
    band = lambda a, b: float(10 * np.log10(spec[(freqs >= a) & (freqs < b)].sum() / total + 1e-12))
    return {"rms_db": float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)),
            "peak": float(np.max(np.abs(audio))),
            "centroid_hz": float((freqs * spec).sum() / total),
            "bass_db": band(20, 250), "mid_db": band(250, 4000), "high_db": band(4000, 20000)}
