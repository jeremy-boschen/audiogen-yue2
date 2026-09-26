#!/usr/bin/env python3
"""Does a knob do the same thing where the voice sings as where it doesn't?

    bin/latent_knobs_sung.py RUN [RUN...] --out DIR [--channels 0,26,34,55]

The vocal voice of each run's own score says when the voice sings (score time
matches audio time on these runs; Burn It Down's is clipped at the 200 s render).
Frames inside a vocal note are "sung"; frames more than 1 s from any vocal note
are "gaps" (the decoder hears ~0.5 s either side, so nearer frames are mixed).

1. Measure: each knob is applied to the whole song, and its EQ curve and width
   change are measured separately over sung and gap frames. The two sets of
   frames also differ in arrangement, so a difference is a lead, not a proof.
2. Listen: each knob is also rendered only on the sung spans and only on the
   gaps, so the ear can tell whether it takes hold of the voice.

Writes DIR/sung.json, DIR/<song>/*.wav and DIR/index.html. Signal statistics,
not a judgement of how anything sounds.
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

from audiogen import explore, latent_mixer as mixer  # noqa: E402
import latent_atlas as atlas  # noqa: E402

RATE = 48000
NAMED = {0: "radio", 15: "width", 26: "spread", 34: "sustain", 55: "air"}
MARGIN = 1.0


def sung_frames(score_text: str, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per latent frame (25/s): inside a vocal note, and more than MARGIN s from any."""
    layer = explore.score_layer(score_text)
    vocal = next(k for k in layer["voices"] if k.lower().startswith("vocal"))
    t = np.arange(frames) / 25
    sung = np.zeros(frames, bool)
    near = np.zeros(frames, bool)
    for t0, dur in zip(layer["voices"][vocal]["t0"], layer["voices"][vocal]["dur"]):
        sung |= (t >= t0) & (t < t0 + dur)
        near |= (t >= t0 - MARGIN) & (t < t0 + dur + MARGIN)
    return sung, ~near


def split_compare(edited: np.ndarray, original: np.ndarray, sung: np.ndarray, gaps: np.ndarray) -> dict:
    a, o = atlas.band_power(edited), atlas.band_power(original)
    centre = (np.arange(len(a)) * atlas.HOP + atlas.N_FFT / 2) / RATE          # STFT frame centres, seconds
    at = np.minimum((centre * 25).astype(int), len(sung) - 1)
    out = {}
    for name, mask in (("sung", sung[at]), ("gaps", gaps[at])):
        if mask.sum() < 3:
            out[name] = None
            continue
        curve = 10 * np.log10(a[mask].mean(axis=0) / o[mask].mean(axis=0))
        sample = np.repeat(mask, atlas.HOP)[:len(edited)]
        sample = np.pad(sample, (0, len(edited) - len(sample)))
        width = lambda x: 10 * np.log10(np.sum(((x[:, 0] - x[:, 1]) / 2) ** 2) / np.sum(x.mean(axis=1) ** 2))
        out[name] = {"curve": curve.round(2).tolist(), "size_db": round(float(np.sqrt(np.mean(curve ** 2))), 2),
                     "width_db": round(float(width(edited[sample]) - width(original[sample])), 2),
                     "seconds": round(float(mask.sum() * atlas.HOP / RATE), 1)}
    if out["sung"] and out["gaps"]:
        out["curve_corr"] = round(float(np.corrcoef(out["sung"]["curve"], out["gaps"]["curve"])[0, 1]), 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--channels", default=",".join(map(str, NAMED)))
    parser.add_argument("--amount", type=float, default=2.0)
    parser.add_argument("--models", default=str(HERE / "models"))
    args = parser.parse_args()
    import microscope
    from audiogen import microscope as scope
    channels = [int(c) for c in args.channels.split(",")]
    out = pathlib.Path(args.out).expanduser()
    songs = []
    pipe = None
    for run in (pathlib.Path(r).expanduser().resolve() for r in args.runs):
        label = run.parent.name if run.name == "on-cpu" else run.name
        if pipe is None:
            song, _ = microscope.song_for_run(run, args.models)
            pipe = microscope.renderer.build_pipeline(pathlib.Path(args.models), song, progress=False)
        latent = np.load(run / "final" / "latent.npy").astype(np.float32)
        whole = mixer.stats(latent)
        sung, gaps = sung_frames((run / "take" / "score.abc").read_text(), len(latent))
        print(f"{label}: {sung.sum() / 25:.0f} s sung, {gaps.sum() / 25:.0f} s of gaps", flush=True)
        original = np.asarray(pipe.decode(latent), np.float32)
        folder = out / label
        renders = {"original": original}
        rows = {}
        for channel in channels:
            edits = mixer.neutral()
            edits["offset"][channel] = args.amount
            edited_latent = mixer.apply(latent, edits, whole)
            everywhere = np.asarray(pipe.decode(edited_latent), np.float32)
            rows[channel] = split_compare(everywhere, original, sung, gaps)
            renders[f"ch{channel}_everywhere"] = everywhere
            for name, mask in (("sung_only", sung), ("gaps_only", ~sung)):
                local = np.where(mask[:, None], edited_latent, latent).astype(np.float32)
                renders[f"ch{channel}_{name}"] = np.asarray(pipe.decode(local), np.float32)
            r = rows[channel]
            show = lambda s: f"{s['size_db']:4.1f} dB w{s['width_db']:+.1f}" if s else "   -"
            print(f"  ch {channel:>2} {NAMED.get(channel, ''):<8} sung {show(r['sung'])}  gaps {show(r['gaps'])}  alike {r.get('curve_corr', '-')}", flush=True)
        gain = 10 ** (-1 / 20) / max(float(np.abs(a).max()) for a in renders.values())   # one gain: levels stay comparable
        for name, audio in renders.items():
            scope.write_wav(folder / f"{name}.wav", np.clip(audio * gain, -1, 1), RATE, subtype="PCM_16")
        spans = [[round(float(a) / 25, 2), round(float(b) / 25, 2)] for a, b in
                 zip(*[np.flatnonzero(np.diff(np.r_[0, sung.astype(int), 0]) == s) for s in (1, -1)])]
        songs.append({"label": label, "seconds": round(len(latent) / 25, 2), "sung_spans": spans,
                      "channels": rows, "renders": sorted(renders)})
    data = {"amount": args.amount, "bands_hz": atlas.CENTRES.round(1).tolist(), "named": NAMED, "margin_s": MARGIN,
            "songs": songs, "note": "sung vs gap frames also differ in arrangement; a difference is a lead, not a proof"}
    (out / "sung.json").write_text(json.dumps(data, indent=1) + "\n")
    (out / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data)))
    print("wrote", out / "index.html")
    return 0


PAGE = """<!doctype html><meta charset="utf-8"><title>Knobs on the voice vs the band</title>
<style>
body{margin:0;background:#0b1020;color:#e8ecf6;font:14px/1.5 -apple-system,system-ui,sans-serif;padding:18px 22px 40px}
h1{font-size:19px;margin:0}h2{font-size:15px;margin:22px 0 6px}p{color:#9aa4bd;max-width:900px;margin:4px 0 12px}
button{font:inherit;color:#e8ecf6;background:#18223d;border:1px solid #26304f;border-radius:9px;padding:5px 11px;margin:2px}
button.on{background:#3b4f94;border-color:#6b82d6}.row{margin:6px 0}.lab{display:inline-block;width:120px;color:#9aa4bd}
.m{color:#9aa4bd;font-size:12px;font-variant-numeric:tabular-nums;margin-left:124px}
#tl{height:14px;background:#121a30;border-radius:4px;position:relative;margin:8px 0;cursor:pointer;max-width:900px}
#tl .s{position:absolute;top:0;bottom:0;background:#6b3f7a}#tl .p{position:absolute;top:-3px;bottom:-3px;width:2px;background:#fff}
</style>
<h1>Does a knob grab the voice?</h1>
<p>Each knob is pushed <b id="amt"></b> of the song's spread: <b>everywhere</b>, only while the voice <b>sings</b>, or only in the <b>gaps</b>.
If "sung only" changes the voice and the band sounds untouched, the knob reaches the voice. The bar shows where the score says the voice sings (purple); click it to jump.
Everything plays at one shared gain and keeps your place when you switch. Space plays or pauses.
The numbers under each knob are signal statistics, sung frames vs gap frames: the arrangement differs between them too, so a difference is a lead, not a proof.</p>
<div id="songs"></div>
<div id="tl"></div>
<div id="rows"></div>
<script>
const D = __DATA__;
document.getElementById('amt').textContent = D.amount + '×';
let song = D.songs[0], cur = null, audio = new Audio();
function src(n) { return `${encodeURIComponent(song.label)}/${n}.wav`; }
function go(n) { const t = audio.currentTime || 0, p = !audio.paused; audio.pause(); audio = new Audio(src(n)); cur = n;
  audio.addEventListener('loadedmetadata', () => { audio.currentTime = Math.min(t, audio.duration - 0.1); if (p) audio.play(); }, {once: true});
  document.querySelectorAll('#rows button').forEach((b) => b.classList.toggle('on', b.dataset.n === n)); }
function pick(s) { song = s; document.querySelectorAll('#songs button').forEach((b) => b.classList.toggle('on', b.textContent === s.label));
  document.getElementById('tl').innerHTML = s.sung_spans.map(([a, b]) => `<div class="s" style="left:${a / s.seconds * 100}%;width:${(b - a) / s.seconds * 100}%"></div>`).join('') + '<div class="p" id="ph"></div>';
  const fmt = (x) => x ? `${x.size_db} dB, width ${x.width_db >= 0 ? '+' : ''}${x.width_db}` : 'n/a';
  document.getElementById('rows').innerHTML = `<div class="row"><span class="lab">untouched</span><button data-n="original">original</button></div>` +
    Object.entries(s.channels).map(([ch, r]) => `<div class="row"><span class="lab">ch ${ch} ${D.named[ch] || ''}</span>` +
      ['everywhere', 'sung_only', 'gaps_only'].map((k) => `<button data-n="ch${ch}_${k}">${k.replace('_', ' ')}</button>`).join('') +
      `</div><div class="m">sung: ${fmt(r.sung)} · gaps: ${fmt(r.gaps)} · curves alike ${r.curve_corr ?? 'n/a'}</div>`).join('');
  document.querySelectorAll('#rows button').forEach((b) => b.onclick = () => go(b.dataset.n)); go(cur && s.renders.includes(cur) ? cur : 'original'); }
document.getElementById('songs').innerHTML = D.songs.map((s) => `<button>${s.label}</button>`).join('');
document.querySelectorAll('#songs button').forEach((b, i) => b.onclick = () => { audio.pause(); audio.currentTime = 0; pick(D.songs[i]); });
document.getElementById('tl').onclick = (e) => { const b = e.currentTarget.getBoundingClientRect(); audio.currentTime = (e.clientX - b.left) / b.width * song.seconds; };
setInterval(() => { const p = document.getElementById('ph'); if (p) p.style.left = (audio.currentTime / song.seconds * 100) + '%'; }, 100);
addEventListener('keydown', (e) => { if (e.key === ' ') { e.preventDefault(); audio.paused ? audio.play() : audio.pause(); } });
pick(song);
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
