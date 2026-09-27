#!/usr/bin/env python3
"""Do the latent channel knobs do the same thing on every song?

    bin/latent_knobs_across.py --runs RUN [RUN...] [--audio FILE...] --out DIR

Each source gives a latent: a microscope run's final latent (what the model made),
or a finished recording put through the VAE encoder. On a 20 s excerpt from each,
every channel is pushed up and down by N of *that song's own* standard deviations
(as in the mixer) and compared with the untouched decode:

  curve     level change per band, dB (the same 24 bands as bin/latent_atlas.py)
  width_db  change in side/mid energy: + wider, - narrower
  level_db  overall level change

Then, per channel, how alike the curves are across songs (correlation with the
first source's curve, and the size ratio). Writes DIR/across.json and index.html,
one card per channel with every song's curve overlaid. Signal statistics, not a
judgement of how anything sounds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "bin"))

from audiogen import latent_mixer as mixer  # noqa: E402
import latent_atlas as atlas  # noqa: E402

RATE = 48000
# What each knob did on Slow Down take 2 (DONE.md): the names the page shows.
NAMED = {0: "radio", 1: "grain", 2: "soft attack", 15: "width", 22: "drive", 26: "spread",
         34: "sustain", 38: "lo-fi", 51: "drive", 55: "air"}


def width_db(audio: np.ndarray) -> float:
    mid, side = audio.mean(axis=1), (audio[:, 0] - audio[:, 1]) / 2
    return float(10 * np.log10((np.sum(side ** 2) + 1e-12) / (np.sum(mid ** 2) + 1e-12)))


def encode_file(encoder, path: pathlib.Path) -> np.ndarray:
    import torch
    import microscope
    with tempfile.TemporaryDirectory() as tmp:
        wav = pathlib.Path(tmp) / "in.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ar", str(RATE), "-ac", "2",
                        "-f", "wav", "-c:a", "pcm_f32le", str(wav)], check=True)
        audio = microscope.decoded_pcm(wav)
    parts, step = [], RATE * 30                                   # 30 s pieces keep encoder memory small
    with torch.inference_mode():
        for a in range(0, len(audio) - 1920, step):
            piece = audio[a:a + step]
            piece = piece[:len(piece) // 1920 * 1920]
            parts.append(encoder.encode(torch.as_tensor(piece.T[None], dtype=torch.float32)).squeeze(0).T.float().cpu().numpy())
    return np.concatenate(parts)


def measure(pipe, latent: np.ndarray, channels: list[int], amount: float, seconds: float, at: float) -> dict:
    whole = mixer.stats(latent)
    start = max(0, min(len(latent) - round(seconds * 25), round(at * len(latent))))
    part = latent[start:start + round(seconds * 25)]
    original = np.asarray(pipe.decode(part), np.float32)
    base_width = width_db(original)
    rows = {}
    for channel in channels:
        row = {}
        for sign, name in ((1, "up"), (-1, "down")):
            edits = mixer.neutral()
            edits["offset"][channel] = sign * amount
            edited = np.asarray(pipe.decode(mixer.apply(part, edits, whole)), np.float32)
            row[name] = {**atlas.compare(edited, original), "width_db": round(width_db(edited) - base_width, 2)}
        rows[channel] = row
    return {"excerpt_start": round(start / 25, 2), "seconds": seconds, "channels": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--audio", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--channels", default="all", help="'all' or a comma list")
    parser.add_argument("--amount", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--at", type=float, default=0.45, help="excerpt position as a fraction of the song")
    parser.add_argument("--models", default=str(HERE / "models"))
    args = parser.parse_args()
    import microscope
    runs = [pathlib.Path(r).expanduser().resolve() for r in args.runs]
    if not runs:
        raise SystemExit("at least one --runs source is needed to build the decoder")
    channels = list(range(mixer.CHANNELS)) if args.channels == "all" else [int(c) for c in args.channels.split(",")]
    song, _ = microscope.song_for_run(runs[0], args.models)
    pipe = microscope.renderer.build_pipeline(pathlib.Path(args.models), song, progress=False)
    sources = []
    for run in runs:
        label = run.parent.name if run.name == "on-cpu" else run.name
        sources.append({"label": label, "kind": "model latent", "path": str(run),
                        "latent": np.load(run / "final" / "latent.npy").astype(np.float32)})
    if args.audio:
        from yue2.modeling_vae import YuE2VAE
        encoder = YuE2VAE.from_pretrained(pathlib.Path(args.models) / "YuE2-Vae", decoder_only=False, device=pipe.device,
                                          local_files_only=True, profile=pipe.profile)
        for path in map(pathlib.Path, args.audio):
            sources.append({"label": path.stem, "kind": "encoded recording", "path": str(path),
                            "latent": encode_file(encoder, path)})
        del encoder
    results = []
    for source in sources:
        print(f"{source['label']} ({source['kind']}, {len(source['latent']) / 25:.0f} s)", flush=True)
        results.append({k: v for k, v in source.items() if k != "latent"}
                       | measure(pipe, source["latent"], channels, args.amount, args.seconds, args.at))
    agreement = {}
    for channel in channels:
        ref = results[0]["channels"][channel]
        row = {}
        for r in results[1:]:
            other = r["channels"][channel]
            row[r["label"]] = {d: {"curve_corr": round(float(np.corrcoef(ref[d]["curve"], other[d]["curve"])[0, 1]), 2),
                                   "size_ratio": round(other[d]["size_db"] / max(ref[d]["size_db"], 0.05), 2),
                                   "width_db": other[d]["width_db"]} for d in ("up", "down")}
        agreement[channel] = row
    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    data = {"amount": args.amount, "bands_hz": atlas.CENTRES.round(1).tolist(), "named": NAMED,
            "sources": results, "agreement": agreement,
            "note": "each song's knob measured against its own untouched decode; agreement is vs the first source"}
    (out / "across.json").write_text(json.dumps(data, indent=1) + "\n")
    (out / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data)))
    ref = results[0]
    print(f"\nagreement with {ref['label']} (curve correlation up/down, size ratio up, width change up):")
    for channel in channels:
        if channel not in NAMED and args.channels == "all":
            continue
        cells = "  ".join(f"{label[:14]:>14} {v['up']['curve_corr']:+.2f}/{v['down']['curve_corr']:+.2f} x{v['up']['size_ratio']:<4} w{v['up']['width_db']:+.1f}"
                          for label, v in agreement[channel].items())
        print(f"  ch {channel:>2} {NAMED.get(channel, ''):<11} ref w{ref['channels'][channel]['up']['width_db']:+.1f}  {cells}")
    print("wrote", out / "index.html")
    return 0


PAGE = """<!doctype html><meta charset="utf-8"><title>Do the knobs hold across songs?</title>
<style>
body{margin:0;background:#0b1020;color:#e8ecf6;font:14px/1.45 -apple-system,system-ui,sans-serif;padding:18px 22px 40px}
h1{font-size:19px;margin:0}p{color:#9aa4bd;margin:4px 0 12px;max-width:1000px}
.bar{display:flex;gap:14px;align-items:center;margin-bottom:14px;color:#9aa4bd;font-size:13px}
select{font:inherit;color:#e8ecf6;background:#18223d;border:1px solid #26304f;border-radius:8px;padding:5px 9px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:10px}
.c{background:#121a30;border:1px solid #26304f;border-radius:12px;padding:8px 10px}
.c h3{margin:0;font-size:13px;display:flex;justify-content:space-between}.c h3 span{color:#9aa4bd;font-weight:400;font-size:11.5px}
.m{color:#9aa4bd;font-size:11px;font-variant-numeric:tabular-nums}svg{display:block;width:100%;height:90px}
.k{display:inline-block;width:12px;height:3px;border-radius:2px;margin:0 4px 2px 10px;vertical-align:middle}
</style>
<h1>Do the knobs hold across songs?</h1>
<p>Each card shows one latent channel pushed <b id="amt"></b> of each song's own spread, drawn as that song's EQ curve (40 Hz on the left, 20 kHz on the right, grid at ±6 dB). If the lines lie on top of each other, the knob does the same thing on every song.
<b>w</b> is the change in stereo width (dB of side vs mid). Signal statistics, not a verdict.</p>
<div class="bar">Direction <select id="dir"><option value="up">pushed up</option><option value="down">pushed down</option></select>
Show <select id="which"><option value="named">named knobs</option><option value="all">all 64</option></select><span id="key"></span></div>
<div class="grid" id="grid"></div>
<script>
const D = __DATA__, COL = ['#ffc2e3', '#8fd3ff', '#b6f0a0', '#ffd98f', '#b69cff', '#ff9f8f'];
document.getElementById('amt').textContent = D.amount + '×';
document.getElementById('key').innerHTML = D.sources.map((s, i) => `<span class="k" style="background:${COL[i]}"></span>${s.label} <span style="color:#5f6a86">(${s.kind})</span>`).join('');
const W = 260, H = 90, R = 18, n = D.bands_hz.length;
const x = (i) => 4 + i / (n - 1) * (W - 8), y = (db) => H / 2 - Math.max(-R, Math.min(R, db)) / R * (H / 2 - 4);
const path = (c) => c.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
function card(ch, dir) {
  const grid = [6, -6].map((d) => `<line x1="0" x2="${W}" y1="${y(d)}" y2="${y(d)}" stroke="#26304f" stroke-dasharray="3 3"/>`).join('') + `<line x1="0" x2="${W}" y1="${y(0)}" y2="${y(0)}" stroke="#3a4670"/>`;
  const lines = D.sources.map((s, i) => `<path d="${path(s.channels[ch][dir].curve)}" fill="none" stroke="${COL[i]}" stroke-width="1.6"/>`).join('');
  const widths = D.sources.map((s, i) => `<span style="color:${COL[i]}">w${s.channels[ch][dir].width_db >= 0 ? '+' : ''}${s.channels[ch][dir].width_db}</span>`).join(' ');
  const corr = Object.values(D.agreement[ch]).map((v) => v[dir].curve_corr.toFixed(2)).join(' ');
  return `<div class="c"><h3>ch ${ch}<span>${D.named[ch] || ''}</span></h3><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${grid}${lines}</svg>
    <div class="m">${widths} · alike ${corr}</div></div>`;
}
function render() { const dir = document.getElementById('dir').value, all = document.getElementById('which').value === 'all';
  const chans = Object.keys(D.sources[0].channels).filter((c) => all || D.named[c]);
  document.getElementById('grid').innerHTML = chans.map((c) => card(c, dir)).join(''); }
['dir', 'which'].forEach((id) => document.getElementById(id).onchange = render); render();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
