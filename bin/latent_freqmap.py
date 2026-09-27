#!/usr/bin/env python3
"""Does a latent channel act only at some frequencies? A channel x frequency map.

    bin/latent_freqmap.py RUN [--amount 2]

One test sound: 16 pure tones stepped from 50 Hz to 10 kHz, half a second each.
It is encoded with the VAE's encoder; each channel is pushed up and down by N of
its standard deviations (the take's, as in the mixer); and for every tone the
level change and the change in harmonic distortion are measured against the
unedited round trip. Writes RUN/latent_freqmap/freqmap.json and index.html
(two heat maps). Signal statistics, not a judgement of how anything sounds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from audiogen import latent_mixer as mixer  # noqa: E402

RATE = 48000
TONES = np.geomspace(50, 10000, 16)
STEP = 0.5
T = np.arange(int(RATE * STEP * len(TONES))) / RATE


def stepped() -> np.ndarray:
    freq = TONES[np.minimum((T / STEP).astype(int), len(TONES) - 1)]
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    x = 0.2 * np.sin(phase)
    return np.stack([x, x], axis=1).astype(np.float32)


def per_tone(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Level of each tone's fundamental (dB) and its harmonics relative to it (dB)."""
    m = audio.mean(axis=1)
    level, dist = [], []
    for i, f0 in enumerate(TONES):
        seg = m[int((i * STEP + 0.08) * RATE):int(((i + 1) * STEP - 0.08) * RATE)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1 / RATE)
        near = lambda f: (freqs > f * 0.97 - 5) & (freqs < f * 1.03 + 5)
        fund = spec[near(f0)].sum() + 1e-20
        harm = sum(spec[near(f0 * k)].sum() for k in range(2, 8) if f0 * k < RATE / 2 - 500)
        level.append(10 * np.log10(fund)); dist.append(10 * np.log10(harm / fund + 1e-12))
    return np.array(level), np.array(dist)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--amount", type=float, default=2.0)
    parser.add_argument("--vae", default=str(HERE / "models" / "YuE2-Vae"))
    args = parser.parse_args()
    import torch
    from yue2.modeling_vae import YuE2VAE
    from yue2.profiles import ComfyUIYuE2MPSProfile
    run = pathlib.Path(args.run).expanduser().resolve()
    whole = mixer.stats(np.load(run / "final" / "latent.npy").astype(np.float32))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vae = YuE2VAE.from_pretrained(args.vae, decoder_only=False, device=device, local_files_only=True,
                                  profile=ComfyUIYuE2MPSProfile())
    with torch.inference_mode():
        z = vae.encode(torch.as_tensor(stepped().T[None])).squeeze(0).T.cpu().numpy()
        decode = lambda lat: vae.decode(torch.as_tensor(lat.T[None], dtype=torch.float32)).squeeze(0).T.cpu().numpy()[:len(T)]
        base_level, base_dist = per_tone(decode(z))
        rows = []
        for channel in range(mixer.CHANNELS):
            row = {"channel": channel}
            for sign, direction in ((1, "up"), (-1, "down")):
                edits = mixer.neutral()
                edits["offset"][channel] = sign * args.amount
                level, dist = per_tone(decode(mixer.apply(z, edits, whole)))
                row[direction] = {"level_db": (level - base_level).round(2).tolist(), "distortion_db": (dist - base_dist).round(2).tolist()}
            rows.append(row)
            u, d = np.array(row["up"]["level_db"]), np.array(row["down"]["level_db"])
            k = int(np.argmax(np.maximum(np.abs(u), np.abs(d))))
            print(f"  ch {channel:>2}  biggest level change {max(abs(u[k]), abs(d[k])):5.2f} dB at {TONES[k]:6.0f} Hz  "
                  f"(spread: {np.std(np.maximum(np.abs(u), np.abs(d))):.2f})", flush=True)
    out = run / "latent_freqmap"
    out.mkdir(exist_ok=True)
    data = {"run": str(run), "amount": args.amount, "tones_hz": TONES.round(1).tolist(),
            "roundtrip_level_db": base_level.round(2).tolist(), "roundtrip_distortion_db": base_dist.round(2).tolist(),
            "channels": rows, "note": "change per tone vs the unedited round trip; signal statistics"}
    (out / "freqmap.json").write_text(json.dumps(data, indent=1) + "\n")
    (out / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data)))
    print("wrote", out / "index.html")
    return 0


PAGE = """<!doctype html><meta charset="utf-8"><title>Latent channels by frequency</title>
<style>
body{margin:0;background:#0b1020;color:#e8ecf6;font:13.5px/1.45 -apple-system,system-ui,sans-serif;padding:18px 22px 40px}
h1{font-size:19px;margin:0}h2{font-size:15px;margin:22px 0 6px}p{color:#9aa4bd;margin:4px 0 12px;max-width:1100px}
.bar{display:flex;gap:12px;align-items:center;color:#9aa4bd;font-size:13px}select{font:inherit;color:#e8ecf6;background:#18223d;border:1px solid #26304f;border-radius:8px;padding:4px 8px}
canvas{display:block;image-rendering:pixelated;margin-top:6px}.key{display:flex;gap:10px;align-items:center;color:#9aa4bd;font-size:12px;margin-top:6px}
.key span{display:inline-block;width:120px;height:10px;border-radius:3px}
#tip{position:fixed;pointer-events:none;background:#121a30;border:1px solid #26304f;border-radius:8px;padding:4px 8px;font-size:12px;display:none}
</style>
<h1>Where in the spectrum each latent channel acts</h1>
<p>Rows are the 64 channels, columns are 16 pure tones from 50 Hz to 10 kHz. Colour is the change at that tone when the channel is pushed by <b id="amt"></b> of its spread:
<b style="color:#ff8fc8">pink</b> louder or more distorted, <b style="color:#8fd3ff">blue</b> quieter or cleaner, dark means no change. A channel that works only in one register shows as a short bright stripe. Hover for numbers. Signal statistics, not a verdict.</p>
<div class="bar">Direction <select id="dir"><option value="up">pushed up</option><option value="down">pushed down</option></select>
Scale <select id="scale"><option value="6">±6 dB</option><option value="3">±3 dB</option><option value="12">±12 dB</option></select>
Order <select id="order"><option value="channel">channel</option><option value="size">biggest change</option></select></div>
<h2>Level change of the tone</h2><canvas id="lv"></canvas>
<h2>Harmonic distortion change</h2><canvas id="ds"></canvas>
<div class="key">blue <span style="background:linear-gradient(90deg,#8fd3ff,#0b1020,#ff8fc8)"></span> pink</div>
<div id="tip"></div>
<script>
const D = __DATA__;
document.getElementById('amt').textContent = D.amount + '×';
const CW = 44, CH = 13, LEFT = 44, TOP = 18;
function colour(v, s) { const t = Math.max(-1, Math.min(1, v / s));
  const [r, g, b] = t >= 0 ? [255, 143, 200] : [143, 211, 255]; const k = Math.abs(t);
  return `rgb(${Math.round(11 + (r - 11) * k)},${Math.round(16 + (g - 16) * k)},${Math.round(32 + (b - 32) * k)})`; }
function order() { const dir = document.getElementById('dir').value, rows = [...D.channels];
  if (document.getElementById('order').value === 'size') rows.sort((a, b) => Math.max(...b[dir].level_db.map(Math.abs)) - Math.max(...a[dir].level_db.map(Math.abs)));
  return rows; }
function draw(id, key) {
  const c = document.getElementById(id), dir = document.getElementById('dir').value, s = +document.getElementById('scale').value, rows = order();
  c.width = LEFT + CW * D.tones_hz.length; c.height = TOP + CH * rows.length; const g = c.getContext('2d');
  g.fillStyle = '#0b1020'; g.fillRect(0, 0, c.width, c.height); g.font = '10px -apple-system,sans-serif'; g.fillStyle = '#5f6a86';
  D.tones_hz.forEach((f, j) => g.fillText(f >= 1000 ? (f / 1000).toFixed(1) + 'k' : Math.round(f), LEFT + j * CW + 6, 12));
  rows.forEach((r, i) => { g.fillStyle = '#9aa4bd'; g.fillText('ch ' + r.channel, 4, TOP + i * CH + 10);
    r[dir][key].forEach((v, j) => { g.fillStyle = colour(v, s); g.fillRect(LEFT + j * CW, TOP + i * CH, CW - 1, CH - 1); }); });
  c.onmousemove = (e) => { const b = c.getBoundingClientRect(), j = Math.floor((e.clientX - b.left - LEFT) / CW), i = Math.floor((e.clientY - b.top - TOP) / CH), tip = document.getElementById('tip');
    if (j < 0 || i < 0 || j >= D.tones_hz.length || i >= rows.length) { tip.style.display = 'none'; return; }
    tip.style.display = 'block'; tip.style.left = e.clientX + 12 + 'px'; tip.style.top = e.clientY + 12 + 'px';
    tip.textContent = `ch ${rows[i].channel} · ${Math.round(D.tones_hz[j])} Hz · ${rows[i][dir][key][j] >= 0 ? '+' : ''}${rows[i][dir][key][j]} dB`; };
  c.onmouseleave = () => { document.getElementById('tip').style.display = 'none'; };
}
function render() { draw('lv', 'level_db'); draw('ds', 'distortion_db'); }
['dir', 'scale', 'order'].forEach((id) => document.getElementById(id).onchange = render); render();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
