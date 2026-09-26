#!/usr/bin/env python3
"""What each of the 64 latent channels does to the sound, measured as an EQ curve.

    bin/latent_atlas.py RUN [--start 20 --seconds 20 --amount 2]

For every channel, decodes the excerpt with that channel offset by +amount and
-amount of its own standard deviations, and compares each against the untouched
decode, band by band (24 log bands, 40 Hz - 20 kHz, each at least 4 FFT bins wide):

  curve      mean level change per band, dB: the channel's EQ curve
  steadiness how much that change moves over time (median over bands of the std
             of the per-frame dB change); low means it acts like a fixed filter
  mirror     correlation of the +amount and -amount curves; near -1 means the
             two directions are opposite settings of one knob

Writes RUN/latent_atlas/atlas.json and index.html. Signal statistics, not a
judgement of how anything sounds.
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
# 24 bands from 40 Hz: with a 16384-point FFT (2.9 Hz resolution) the narrowest band
# still holds 4 bins. 32 bands from 30 Hz on a 4096-point FFT left some bands with
# one bin or none, which drew spikes (or a flat 0) at the bottom of every curve.
N_FFT, HOP = 16384, 4096
EDGES = np.geomspace(40, 20000, 25)
CENTRES = np.sqrt(EDGES[:-1] * EDGES[1:])


def band_power(audio: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """[frames, bands] power per log band of the mono mix."""
    x = audio.mean(axis=1)
    window = np.hanning(n_fft)
    frames = np.stack([x[i:i + n_fft] * window for i in range(0, len(x) - n_fft, hop)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / RATE)
    which = np.digitize(freqs, EDGES) - 1
    if min(int((which == b).sum()) for b in range(len(CENTRES))) < 4:
        raise ValueError("a band holds fewer than 4 FFT bins; widen the bands or lengthen the FFT")
    return np.stack([power[:, which == b].sum(axis=1) for b in range(len(CENTRES))], axis=1) + 1e-12


def compare(edited: np.ndarray, original: np.ndarray) -> dict:
    a, o = band_power(edited), band_power(original)
    per_frame = 10 * np.log10(a / o)
    curve = 10 * np.log10(a.mean(axis=0) / o.mean(axis=0))
    level = 10 * np.log10(a.sum() / o.sum())
    return {"curve": curve.round(2).tolist(), "level_db": round(float(level), 2),
            "steadiness_db": round(float(np.median(per_frame.std(axis=0))), 2),
            "size_db": round(float(np.sqrt(np.mean(curve ** 2))), 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--start", type=float, default=20)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--amount", type=float, default=2.0)
    parser.add_argument("--gain", type=float, default=3.0, help="the mixer's gain slider maximum")
    parser.add_argument("--models", default=str(HERE / "models"))
    args = parser.parse_args()
    import microscope
    run = pathlib.Path(args.run).expanduser().resolve()
    latent = np.load(run / "final" / "latent.npy").astype(np.float32)
    whole = mixer.stats(latent)
    song, _ = microscope.song_for_run(run, args.models)
    pipe = microscope.renderer.build_pipeline(pathlib.Path(args.models), song, progress=False)
    part = latent[round(args.start * 25):round((args.start + args.seconds) * 25)]
    original = np.asarray(pipe.decode(part), np.float32)
    rows = []
    for channel in range(mixer.CHANNELS):
        row = {"channel": channel}
        for sign, name in ((1, "up"), (-1, "down")):
            edits = mixer.neutral()
            edits["offset"][channel] = sign * args.amount
            row[name] = compare(np.asarray(pipe.decode(mixer.apply(part, edits, whole)), np.float32), original)
        edits = mixer.neutral()
        edits["gain"][channel] = args.gain
        row["gain"] = compare(np.asarray(pipe.decode(mixer.apply(part, edits, whole)), np.float32), original)
        row["mirror"] = round(float(np.corrcoef(row["up"]["curve"], row["down"]["curve"])[0, 1]), 3)
        rows.append(row)
        print(f"  ch {channel:>2}  up {row['up']['size_db']:5.2f} dB  down {row['down']['size_db']:5.2f} dB  "
              f"gain {row['gain']['size_db']:5.2f} dB  steady {row['up']['steadiness_db']:.2f}  mirror {row['mirror']:+.2f}", flush=True)
    out = run / "latent_atlas"
    out.mkdir(exist_ok=True)
    atlas = {"run": str(run), "start": args.start, "seconds": args.seconds, "amount": args.amount, "gain": args.gain,
             "bands_hz": CENTRES.round(1).tolist(), "channels": rows,
             "note": "signal statistics of decoded audio; each curve is the level change per band vs the untouched decode"}
    (out / "atlas.json").write_text(json.dumps(atlas, indent=1) + "\n")
    (out / "index.html").write_text(PAGE.replace("__ATLAS__", json.dumps(atlas)))
    print("wrote", out / "index.html")
    return 0


PAGE = """<!doctype html><meta charset="utf-8"><title>Latent channel atlas</title>
<style>
body{margin:0;background:#0b1020;color:#e8ecf6;font:14px/1.45 -apple-system,system-ui,sans-serif;padding:18px 22px 40px}
h1{font-size:19px;margin:0}p{color:#9aa4bd;margin:4px 0 12px;max-width:980px}
.bar{display:flex;gap:14px;align-items:center;margin-bottom:14px;color:#9aa4bd;font-size:13px}
select,button{font:inherit;color:#e8ecf6;background:#18223d;border:1px solid #26304f;border-radius:8px;padding:5px 9px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.c{background:#121a30;border:1px solid #26304f;border-radius:12px;padding:8px 10px}
.c h3{margin:0;font-size:13px;display:flex;justify-content:space-between}.c h3 span{color:#9aa4bd;font-weight:400;font-size:11.5px}
.c .m{color:#9aa4bd;font-size:11px;font-variant-numeric:tabular-nums}svg{display:block;width:100%;height:84px}
.k{display:inline-block;width:10px;height:3px;border-radius:2px;margin:0 4px 2px 0;vertical-align:middle}
</style>
<h1>What each latent channel does, as an EQ curve</h1>
<p>Each card: the level change per frequency band when that channel is pushed up (<span class="k" style="background:#ffc2e3"></span>pink) or down (<span class="k" style="background:#8fd3ff"></span>blue) by <b id="amt"></b> of its own spread, and with its gain at <b id="gn"></b> (<span class="k" style="background:#b69cff"></span>violet, the mixer's gain slider at its top), against the untouched decode.
Grid lines at ±6 dB; 40 Hz on the left, 20 kHz on the right. <b>steady</b> is how much the change wobbles over time (low = acts like a fixed EQ); <b>mirror</b> near −1 means up and down are opposite settings of one knob.
Signal statistics, not a verdict.</p>
<div class="bar">Sort <select id="sort"><option value="channel">channel</option><option value="size">biggest change</option><option value="gain">biggest gain change</option><option value="steady">most EQ-like</option><option value="mirror">most knob-like</option></select>
<span id="info"></span></div>
<div class="grid" id="grid"></div>
<script>
const A = __ATLAS__;
document.getElementById('amt').textContent = A.amount + '×'; document.getElementById('gn').textContent = A.gain;
document.getElementById('info').textContent = `${A.run.split('/').slice(-2).join('/')} · excerpt ${A.start}–${A.start + A.seconds} s`;
const W = 220, H = 84, R = 18;
const x = (i) => 4 + i / (A.bands_hz.length - 1) * (W - 8), y = (db) => H / 2 - Math.max(-R, Math.min(R, db)) / R * (H / 2 - 4);
const path = (c) => c.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
const tick = (hz) => { const i = A.bands_hz.findIndex((b) => b >= hz); return i < 0 ? null : x(i); };
function card(r) {
  const grid = [6, -6].map((d) => `<line x1="0" x2="${W}" y1="${y(d)}" y2="${y(d)}" stroke="#26304f" stroke-dasharray="3 3"/>`).join('')
    + `<line x1="0" x2="${W}" y1="${y(0)}" y2="${y(0)}" stroke="#3a4670"/>`
    + [100, 1000, 10000].map((hz) => `<line x1="${tick(hz)}" x2="${tick(hz)}" y1="0" y2="${H}" stroke="#1c2440"/><text x="${tick(hz) + 2}" y="${H - 3}" fill="#5f6a86" font-size="9">${hz >= 1000 ? hz / 1000 + 'k' : hz}</text>`).join('');
  return `<div class="c"><h3>ch ${r.channel}<span>up ${r.up.size_db.toFixed(1)} · down ${r.down.size_db.toFixed(1)} · gain ${r.gain.size_db.toFixed(1)} dB</span></h3>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${grid}<path d="${path(r.gain.curve)}" fill="none" stroke="#b69cff" stroke-width="1.6" stroke-dasharray="4 2"/><path d="${path(r.down.curve)}" fill="none" stroke="#8fd3ff" stroke-width="1.6"/><path d="${path(r.up.curve)}" fill="none" stroke="#ffc2e3" stroke-width="1.6"/></svg>
    <div class="m">level ${r.up.level_db >= 0 ? '+' : ''}${r.up.level_db} / ${r.down.level_db >= 0 ? '+' : ''}${r.down.level_db} dB · steady ${r.up.steadiness_db} · mirror ${r.mirror >= 0 ? '+' : ''}${r.mirror}</div></div>`;
}
function render() {
  const k = document.getElementById('sort').value, rows = [...A.channels];
  const size = (r) => r.up.size_db + r.down.size_db;
  if (k === 'gain') rows.sort((a, b) => b.gain.size_db - a.gain.size_db);
  if (k === 'size') rows.sort((a, b) => size(b) - size(a));
  if (k === 'steady') rows.sort((a, b) => (a.up.steadiness_db + a.down.steadiness_db) / Math.max(size(a), 0.1) - (b.up.steadiness_db + b.down.steadiness_db) / Math.max(size(b), 0.1));
  if (k === 'mirror') rows.sort((a, b) => a.mirror - b.mirror);
  document.getElementById('grid').innerHTML = rows.map(card).join('');
}
document.getElementById('sort').onchange = render; render();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
