#!/usr/bin/env python3
"""Could a take's first N seconds be rendered on their own, as a fast preview?

    bin/preview_prefix.py RUN [--seconds 10,30]

For each N, against RUN's own full take:
  semantic  generate only N s of tokens (same plan, sampling and seed, budget cut
            to N*25) and compare with the take's first N*25 tokens
  sound     run the sound stage on the take's first N*25 tokens alone and compare
            its latent and audio with the full take's first N seconds
Times both. Writes RUN/preview_prefix/ (N s preview, the take's first N s, and
index.html to A/B them) and preview.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "bin"))

import microscope  # noqa: E402
from audiogen import microscope as scope  # noqa: E402

RATE = 48000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--seconds", default="10,30")
    parser.add_argument("--models", default=str(HERE / "models"))
    args = parser.parse_args()
    from yue2.pipeline import SemanticResult
    run = pathlib.Path(args.run).expanduser().resolve()
    song, _ = microscope.song_for_run(run, args.models)
    pipe = microscope.renderer.build_pipeline(pathlib.Path(args.models), song, progress=False)
    plan, tokens = microscope.load_take(run)
    config = json.loads((run / "take" / "config.json").read_text())
    sampling = config["overrides"]["semantic"]
    take_latent = np.load(run / "final" / "latent.npy").astype(np.float32)
    take_audio = microscope.decoded_pcm(run / "final" / "audio.wav")
    out = run / "preview_prefix"
    rows = []
    for seconds in (float(s) for s in args.seconds.split(",")):
        n = int(seconds * 25)
        clock = time.perf_counter()
        short = pipe.generate_semantic(plan, sampling={**sampling, "max_tokens": n})
        semantic_s = time.perf_counter() - clock
        got = list(short.tokens)
        same_upto = next((i for i, (a, b) in enumerate(zip(got, tokens)) if a != b), min(len(got), len(tokens)))
        clock = time.perf_counter()
        latent = np.asarray(pipe.synthesize(SemanticResult(plan, tokens[:n], {}, False)), np.float32)
        sound_s = time.perf_counter() - clock
        clock = time.perf_counter()
        audio = np.asarray(pipe.decode(latent), np.float32)
        decode_s = time.perf_counter() - clock
        ref = take_audio[:len(audio)]
        diff = np.abs(latent - take_latent[:n])
        row = {"seconds": seconds, "tokens": n,
               "semantic": {"generated": len(got), "same_as_take_for": same_upto,
                            "identical": got == tokens[:len(got)], "time_s": round(semantic_s, 1)},
               "sound": {"latent_identical": bool(np.array_equal(latent, take_latent[:n])),
                         "latent_max_diff": round(float(diff.max()), 4),
                         "first_diff_frame": int(np.argmax(diff.max(axis=1) > 0)) if diff.max() > 0 else None,
                         "latent_diff_rel_db": round(float(20 * np.log10(np.sqrt(np.mean(diff ** 2)) / take_latent[:n].std())), 1),
                         "audio_diff_rel_db": round(float(20 * np.log10(np.sqrt(np.mean((audio - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))), 1),
                         "envelope_vs_take": {k: round(v, 3) for k, v in scope.envelope_similarity(audio, ref, RATE).items()},
                         "time_s": round(sound_s, 1)},
               "decode_s": round(decode_s, 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        tag = f"{int(seconds)}s"
        scope.write_wav(out / f"preview_{tag}.wav", audio, RATE)
        gain = 10 ** (-1 / 20) / max(np.abs(audio).max(), np.abs(ref).max())
        scope.write_wav(out / "listening" / f"preview_{tag}.wav", np.clip(audio * gain, -1, 1), RATE, subtype="PCM_16")
        scope.write_wav(out / "listening" / f"take_{tag}.wav", np.clip(ref * gain, -1, 1), RATE, subtype="PCM_16")
    (out / "preview.json").write_text(json.dumps({"run": str(run), "rows": rows}, indent=1) + "\n")
    tracks = [(f"take {int(r['seconds'])}s", f"listening/take_{int(r['seconds'])}s.wav") for r in rows] + \
             [(f"preview {int(r['seconds'])}s", f"listening/preview_{int(r['seconds'])}s.wav") for r in rows]
    (out / "index.html").write_text(PAGE.replace("__TRACKS__", json.dumps(tracks)).replace("__ROWS__", json.dumps(rows)))
    print("wrote", out / "index.html")
    return 0


PAGE = """<!doctype html><meta charset="utf-8"><title>Preview from the first seconds</title>
<style>body{margin:0;background:#0b1020;color:#e8ecf6;font:14px/1.5 -apple-system,system-ui,sans-serif;padding:22px}
p{color:#9aa4bd;max-width:780px}button{font:inherit;color:#e8ecf6;background:#18223d;border:1px solid #26304f;border-radius:10px;padding:7px 14px;margin:3px}
button.on{background:#3b4f94;border-color:#6b82d6}pre{color:#9aa4bd;font-size:12px}</style>
<h1 style="font-size:19px">Preview: the first seconds rendered on their own</h1>
<p><b>take</b> is the first N seconds of the full take. <b>preview</b> is the sound stage run on only those N seconds of the song's tokens. Both play at one shared gain, and switching keeps your place. Space plays or pauses.</p>
<div id="b"></div><pre id="rows"></pre>
<script>
const T = __TRACKS__; let cur = 0, el = T.map(([, s]) => { const a = new Audio(s); a.preload = 'auto'; return a; });
T.forEach(([name], i) => { const b = document.createElement('button'); b.textContent = name; b.onclick = () => go(i); document.getElementById('b').append(b); });
function go(i) { const t = el[cur].currentTime, p = !el[cur].paused; el[cur].pause(); cur = i; el[i].currentTime = Math.min(t, (el[i].duration || 1e9) - 0.05); if (p) el[i].play();
  document.querySelectorAll('button').forEach((b, j) => b.classList.toggle('on', j === i)); }
addEventListener('keydown', (e) => { if (e.key === ' ') { e.preventDefault(); el[cur].paused ? el[cur].play() : el[cur].pause(); } });
document.getElementById('rows').textContent = JSON.stringify(__ROWS__, null, 1); go(0);
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
