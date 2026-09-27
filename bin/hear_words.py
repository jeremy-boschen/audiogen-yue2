#!/usr/bin/env python3
"""Which words can be heard in a take, and when: speech recognition on the finished song.

    bin/hear_words.py RUN [--audiocpp ~/dev/ai/audio.cpp]

Cuts RUN/final/audio.wav at the plan's sections (analysis/alignment.json), each with PAD
seconds either side, and on each piece runs audio.cpp's Qwen3-ASR (robust on singing) for
the words, then Qwen3-ForcedAligner for where each starts and ends. A word is kept by the
section its middle falls in. Writes RUN/analysis/heard.json.

Section by section, not the whole song at once: over the full 93 s of Slow Down take 2 the
aligner lost its place in the long instrumental gaps, packing words into 0.08 s runs (it put
the verse's first "My" 3 s early, in the chorus); per section it put it on the pickup note.

These are words a recogniser heard in the whole mix, not the lyrics the song was given:
it can mishear, and the aligner can misplace a word. The page labels them "heard".
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import tempfile

RATE = 16000                    # the aligner reports sample offsets at 16 kHz
PAD = 1.0
ASR = ("qwen3_asr", "Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf")
ALIGNER = ("qwen3_forced_aligner", "Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf")


def run_cli(cli: pathlib.Path, models: pathlib.Path, task: str, spec, audio: pathlib.Path, *extra) -> None:
    family, weights = spec
    subprocess.run([str(cli), "--task", task, "--family", family, "--model", str(models / weights),
                    "--backend", "metal", "--audio", str(audio), "--language", "en", *extra],
                   check=True, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--audiocpp", default=os.environ.get("AUDIOCPP", str(pathlib.Path.home() / "dev/ai/audio.cpp")))
    args = parser.parse_args()
    run, root = pathlib.Path(args.run).resolve(), pathlib.Path(args.audiocpp).expanduser()
    cli, models = root / "build/macos-metal-release/bin/audiocpp_cli", root / "models"
    audio = run / "final/audio.wav"
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     str(audio)], check=True, capture_output=True, text=True).stdout)
    alignment = json.loads((run / "analysis/alignment.json").read_text())
    starts = [s["start_seconds"] for s in alignment["sections"] if s.get("start_seconds") is not None] or [0.0]
    starts[0] = 0.0                                   # anything before the first section is still heard
    words, transcript = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for start, end in zip(starts, starts[1:] + [duration]):
            lo, hi = max(0.0, start - PAD), min(duration, end + PAD)
            clip, text_out, words_out = (pathlib.Path(tmp) / n for n in ("clip.wav", "asr.txt", "words.json"))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(lo), "-to", str(hi), "-i", str(audio), str(clip)],
                           check=True)
            run_cli(cli, models, "asr", ASR, clip, "--text-out", str(text_out))
            text = text_out.read_text().strip()
            if not text:
                continue
            run_cli(cli, models, "align", ALIGNER, clip, "--text", text, "--words-out", str(words_out))
            found = json.loads(words_out.read_text())
            found = found.get("words", found) if isinstance(found, dict) else found
            kept = [{"word": w["word"], "start": round(lo + w["start_sample"] / RATE, 2),
                     "end": round(lo + w["end_sample"] / RATE, 2)} for w in found]
            kept = [w for w in kept if start <= (w["start"] + w["end"]) / 2 < end]
            words += kept
            transcript.append(" ".join(w["word"] for w in kept))
    transcript = " / ".join(transcript)
    out = run / "analysis/heard.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"source": f"{ASR[1].split('/')[0].removesuffix('-GGUF')} for the words, "
                                         f"{ALIGNER[1].split('/')[0].removesuffix('-GGUF')} for their times "
                                         "(audio.cpp), section by section on the finished take",
                               "transcript": transcript, "words": words}, indent=1))
    print(f"wrote {out}: {len(words)} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
