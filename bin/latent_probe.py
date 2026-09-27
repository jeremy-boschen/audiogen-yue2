#!/usr/bin/env python3
"""What each latent channel does to known test sounds: the gear-measurement view.

    bin/latent_probe.py RUN [--amount 2]

Synthesises test signals, encodes them with the VAE's own encoder, pushes one
channel at a time up and down by N of its standard deviations (the take's, as
in the mixer), decodes, and measures the change against the UNEDITED round trip
of the same signal (encode -> decode is not transparent, and a sine is not
music, so the original file is never the reference).

  sine    440 Hz steady      distortion (harmonics), noise floor, pitch
  pluck   220 Hz, decaying   attack time, decay time
  drums   kick + noise hit   crest (punch), attack time, tail after the hit
  left    330 Hz, left only  leakage into the right channel
  wide    independent L/R    stereo width (L/R correlation)
  sweep   20 Hz -> 20 kHz    EQ curve, cleaner than measuring through a song

Writes RUN/latent_probe/probe.json, index.html and the round-trip audio.
Signal statistics, not a judgement of how anything sounds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "bin"))

from audiogen import latent_mixer as mixer  # noqa: E402

RATE = 48000
SECONDS = 8.0
T = np.arange(int(RATE * SECONDS)) / RATE
PLUCK_EVERY, DRUM_EVERY = 2.0, 0.5


def stereo(x, right=None):
    return np.stack([x, x if right is None else right], axis=1).astype(np.float32)


def signals() -> dict:
    rng = np.random.default_rng(7)
    phase = (T % PLUCK_EVERY)
    pluck = sum(np.sin(2 * np.pi * 220 * k * T) / k for k in range(1, 9)) * np.exp(-phase / 0.6) * 0.18
    d = T % DRUM_EVERY
    kick = np.sin(2 * np.pi * (45 * d + 60 * 0.04 * (1 - np.exp(-d / 0.04)))) * np.exp(-d / 0.15)
    hat = rng.normal(size=len(T)) * np.exp(-d / 0.03) * 0.3
    sweep_phase = 2 * np.pi * 20 * SECONDS / np.log(1000) * (np.exp(T / SECONDS * np.log(1000)) - 1)
    return {
        "sine": stereo(0.25 * np.sin(2 * np.pi * 440 * T)),
        "pluck": stereo(pluck),
        "drums": stereo(0.5 * kick + hat),
        "left": stereo(0.25 * np.sin(2 * np.pi * 330 * T), np.zeros_like(T)),
        "wide": stereo(rng.normal(size=len(T)) * 0.1, rng.normal(size=len(T)) * 0.1),
        "sweep": stereo(0.25 * np.sin(sweep_phase)),
    }


# --- measures ----------------------------------------------------------------------------
def envelope(x, window=240):
    m = np.abs(x).mean(axis=1) if x.ndim == 2 else np.abs(x)
    return np.convolve(m, np.ones(window) / window, mode="same")


def hits(every):
    return [int(t * RATE) for t in np.arange(every, SECONDS - every, every)]    # skip the edges


def attack_ms(env, onsets, span=0.1):
    out = []
    for a in onsets:
        seg = env[a:a + int(span * RATE)]
        peak = seg.max() or 1e-9
        lo, hi = np.argmax(seg >= 0.1 * peak), np.argmax(seg >= 0.9 * peak)
        out.append((hi - lo) / RATE * 1000)
    return float(np.median(out))


def decay_ms(env, onsets, every):
    out = []
    for a in onsets:
        seg = env[a:a + int(every * RATE * 0.95)]
        p = int(np.argmax(seg)); after = seg[p:]
        below = np.nonzero(after < seg[p] * 0.1)[0]                          # -20 dB
        out.append((below[0] if len(below) else len(after)) / RATE * 1000)
    return float(np.median(out))


def tail_db(x, onsets):
    m = x.mean(axis=1)
    head = [np.sum(m[a:a + int(0.05 * RATE)] ** 2) for a in onsets]
    tail = [np.sum(m[a + int(0.1 * RATE):a + int(0.4 * RATE)] ** 2) for a in onsets]
    return float(10 * np.log10(np.sum(tail) / np.sum(head) + 1e-12))


def sine_stats(x, f0=440.0):
    m = x.mean(axis=1)[int(2 * RATE):int(6 * RATE)]
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    freqs = np.fft.rfftfreq(len(m), 1 / RATE)
    near = lambda f: (freqs > f - 6) & (freqs < f + 6)
    fund = spec[near(f0)].sum()
    harm = sum(spec[near(f0 * k)].sum() for k in range(2, 11))
    rest = spec.sum() - fund - harm
    k = int(np.argmax(np.where(near(f0), spec, 0)))
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-20)
    pitch = freqs[k] + (a - c) / (2 * (a - 2 * b + c)) * (freqs[1] - freqs[0])
    return {"distortion_db": float(10 * np.log10(harm / fund + 1e-12)),
            "noise_db": float(10 * np.log10(rest / fund + 1e-12)),
            "pitch_cents": float(1200 * np.log2(pitch / f0))}


EDGES = np.geomspace(40, 20000, 25)


def sweep_curve(x):
    """Level per log band of the sweep's decoded spectrum (dB); compared as a difference."""
    m = x.mean(axis=1)
    spec = np.abs(np.fft.rfft(m)) ** 2
    freqs = np.fft.rfftfreq(len(m), 1 / RATE)
    return np.array([10 * np.log10(spec[(freqs >= a) & (freqs < b)].mean() + 1e-20) for a, b in zip(EDGES, EDGES[1:])])


def measure(name, x) -> dict:
    if name == "sine":
        return sine_stats(x)
    if name == "pluck":
        env = envelope(x); on = hits(PLUCK_EVERY)
        return {"attack_ms": attack_ms(env, on), "decay_ms": decay_ms(env, on, PLUCK_EVERY)}
    if name == "drums":
        on = hits(DRUM_EVERY); env = envelope(x, 48)
        m = x.mean(axis=1)
        return {"crest_db": float(20 * np.log10(np.abs(m).max() / (np.sqrt(np.mean(m ** 2)) + 1e-12))),
                "hit_attack_ms": attack_ms(env, on, 0.05), "tail_db": tail_db(x, on)}
    if name == "left":
        return {"leak_db": float(10 * np.log10(np.sum(x[:, 1] ** 2) / (np.sum(x[:, 0] ** 2) + 1e-12) + 1e-12))}
    if name == "wide":
        return {"lr_correlation": float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])}
    if name == "sweep":
        return {"curve": sweep_curve(x)}
    raise KeyError(name)


def delta(edited: dict, base: dict) -> dict:
    out = {}
    for k, v in edited.items():
        if k == "curve":
            d = v - base[k]
            out["eq_curve"] = d.round(2).tolist()
            out["eq_size_db"] = round(float(np.sqrt(np.mean(d ** 2))), 2)
        else:
            out[k] = round(float(v - base[k]), 3)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--amount", type=float, default=2.0)
    parser.add_argument("--vae", default=str(HERE / "models" / "YuE2-Vae"))
    args = parser.parse_args()
    import torch
    from yue2.modeling_vae import YuE2VAE
    from yue2.profiles import ComfyUIYuE2MPSProfile
    import soundfile
    run = pathlib.Path(args.run).expanduser().resolve()
    whole = mixer.stats(np.load(run / "final" / "latent.npy").astype(np.float32))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vae = YuE2VAE.from_pretrained(args.vae, decoder_only=False, device=device, local_files_only=True,
                                  profile=ComfyUIYuE2MPSProfile())
    out = run / "latent_probe"
    (out / "audio").mkdir(parents=True, exist_ok=True)

    def encode(x):
        with torch.inference_mode():
            return vae.encode(torch.as_tensor(x.T[None])).squeeze(0).T.cpu().numpy()

    def decode(z):
        with torch.inference_mode():
            return vae.decode(torch.as_tensor(z.T[None], dtype=torch.float32)).squeeze(0).T.cpu().numpy()[:len(T)]

    tests = signals()
    latents, bases = {}, {}
    for name, x in tests.items():
        latents[name] = encode(x)
        y = decode(latents[name])
        soundfile.write(out / "audio" / f"{name}.source.wav", x, RATE, subtype="FLOAT")
        soundfile.write(out / "audio" / f"{name}.roundtrip.wav", y, RATE, subtype="FLOAT")
        bases[name] = measure(name, y)
    rows = []
    for channel in range(mixer.CHANNELS):
        row = {"channel": channel}
        for sign, direction in ((1, "up"), (-1, "down")):
            edits = mixer.neutral()
            edits["offset"][channel] = sign * args.amount
            row[direction] = {}
            for name in tests:
                row[direction].update(delta(measure(name, decode(mixer.apply(latents[name], edits, whole))), bases[name]))
        rows.append(row)
        u = row["up"]
        print(f"  ch {channel:>2}  dist {u['distortion_db']:+6.2f}  noise {u['noise_db']:+6.2f}  pitch {u['pitch_cents']:+6.1f}c  "
              f"attack {u['attack_ms']:+6.1f}ms  decay {u['decay_ms']:+7.1f}ms  crest {u['crest_db']:+5.2f}  tail {u['tail_db']:+5.2f}  "
              f"leak {u['leak_db']:+6.2f}  width {u['lr_correlation']:+.3f}  eq {u['eq_size_db']:.2f}", flush=True)
    base_row = {k: {m: (v.round(2).tolist() if isinstance(v, np.ndarray) else round(float(v), 3)) for m, v in b.items()}
                for k, b in bases.items()}
    probe = {"run": str(run), "amount": args.amount, "bands_hz": np.sqrt(EDGES[:-1] * EDGES[1:]).round(1).tolist(),
             "roundtrip": base_row, "channels": rows,
             "note": "changes vs the unedited encode->decode round trip of each test signal; signal statistics"}
    (out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n")
    (out / "index.html").write_text(PAGE.replace("__PROBE__", json.dumps(probe)))
    print("wrote", out / "index.html")
    return 0


PAGE = """<!doctype html><meta charset="utf-8"><title>Latent channel probe</title>
<style>
body{margin:0;background:#0b1020;color:#e8ecf6;font:13.5px/1.45 -apple-system,system-ui,sans-serif;padding:18px 22px 40px}
h1{font-size:19px;margin:0}p{color:#9aa4bd;margin:4px 0 12px;max-width:1100px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:4px 9px;text-align:right;border-bottom:1px solid #1c2440}
th{position:sticky;top:0;background:#121a30;color:#9aa4bd;font-weight:500;cursor:pointer;white-space:nowrap}th:hover{color:#e8ecf6}
th small{display:block;color:#5f6a86;font-weight:400}td:first-child,th:first-child{text-align:left}
.dir{color:#5f6a86;font-size:11px}
</style>
<h1>What each latent channel does to test sounds</h1>
<p>Each channel pushed up and down by <b id="amt"></b> of its spread, measured on synthetic test sounds encoded and decoded by the VAE, against the unedited round trip of the same sound.
Cell colour shows the size of the change within its column; click a heading to sort by the biggest change. Up and down are shown as <span class="dir">up / down</span>. Signal statistics, not a verdict.</p>
<table id="t"></table>
<script>
const P = __PROBE__;
document.getElementById('amt').textContent = P.amount + '×';
const COLS = [
  ['distortion_db', 'distortion', 'dB, sine harmonics'], ['noise_db', 'noise floor', 'dB, sine'], ['pitch_cents', 'pitch', 'cents, sine'],
  ['attack_ms', 'attack', 'ms, pluck'], ['decay_ms', 'decay', 'ms, pluck'], ['crest_db', 'punch', 'dB crest, drums'],
  ['hit_attack_ms', 'hit attack', 'ms, drums'], ['tail_db', 'tail', 'dB after hit'], ['leak_db', 'L→R leak', 'dB, left tone'],
  ['lr_correlation', 'width', 'L/R corr, +narrower'], ['eq_size_db', 'EQ change', 'dB rms, sweep']];
const big = {};
for (const [k] of COLS) big[k] = Math.max(1e-9, ...P.channels.map((r) => Math.max(Math.abs(r.up[k]), Math.abs(r.down[k]))));
const fmt = (v) => (v >= 0 ? '+' : '') + (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
const cell = (r, k) => { const m = Math.max(Math.abs(r.up[k]), Math.abs(r.down[k])) / big[k];
  return `<td style="background:rgba(155,107,255,${(m * 0.75).toFixed(2)})">${fmt(r.up[k])} <span class="dir">/ ${fmt(r.down[k])}</span></td>`; };
let rows = [...P.channels];
function render() {
  document.getElementById('t').innerHTML = `<tr><th data-k="channel">ch</th>${COLS.map(([k, n, s]) => `<th data-k="${k}">${n}<small>${s}</small></th>`).join('')}</tr>`
    + rows.map((r) => `<tr><td>ch ${r.channel}</td>${COLS.map(([k]) => cell(r, k)).join('')}</tr>`).join('');
  document.querySelectorAll('th').forEach((th) => th.onclick = () => { const k = th.dataset.k;
    rows.sort(k === 'channel' ? (a, b) => a.channel - b.channel : (a, b) => Math.max(Math.abs(b.up[k]), Math.abs(b.down[k])) - Math.max(Math.abs(a.up[k]), Math.abs(a.down[k]))); render(); });
}
render();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
