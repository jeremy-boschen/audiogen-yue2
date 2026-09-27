#!/usr/bin/env python3
"""When each audible part of a finished take first comes in.

    bin/find_entries.py RUN [--demucs-python ~/dev/ai/demucs/.venv/bin/python] [--device cpu]

Writes RUN/analysis/entries.json for the demo's "I hear ..." buttons, sorted by entry time.

Instrument parts come from htdemucs_6s stems of RUN/final/audio.wav (demucs runs in its own
venv; stems are cached under RUN/analysis/stems/ and used only to detect entries, never heard).
A stem's entry is the start of the first stretch, at least HOLD s long, in which its 50 ms
frame RMS keeps returning above ENTRY_DB re that stem's own 95th percentile with no gap longer
than GAP s. The gap tolerance is needed because drums here are single hits ~1.35 s apart with
a -60 dB floor between them, and the voice drops between syllables; a strict 1 s hold never
fired for the drums and put the voice 0.65 s late. A stem whose overall level is more than BLEED_DB below the
mix is treated as bleed and left out.

Words and chords reuse existing analysis: "words" is the first word heard.json's recogniser +
aligner placed; "chords" is the first non-N chord of SheetSage on the finished state.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

import numpy as np
import soundfile as sf

MODEL = "htdemucs_6s"
HOP = 0.05          # s per RMS frame
ENTRY_DB = -20.0    # relative to the stem's own 95th percentile
HOLD = 1.0          # s the stem must stay active
GAP = 1.5           # s of dips allowed inside an active stretch
BLEED_DB = -30.0    # stem overall level vs mix below this = bleed
STEMS = {  # stem -> (id, label)
    "drums": ("drums", "the drums"), "bass": ("bass", "the bass"),
    "vocals": ("voice", "a voice"), "guitar": ("guitar", "the guitar"),
    "piano": ("piano", "the piano"), "other": ("other", "the other instruments"),
}


def mono(path: pathlib.Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-10))


def frame_rms(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(HOP * sr)
    k = len(x) // n
    return np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(axis=1))


def onset(rms_db: np.ndarray) -> float | None:
    """Start of the first active stretch (see module doc) lasting at least HOLD s."""
    t = np.flatnonzero(rms_db >= np.percentile(rms_db, 95) + ENTRY_DB) * HOP
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or t[i] - t[i - 1] > GAP:
            if t[i - 1] - t[start] >= HOLD:
                return round(float(t[start]), 2)
            start = i
    return None


def separate(audio: pathlib.Path, out: pathlib.Path, py: str, device: str) -> pathlib.Path:
    stem_dir = out / MODEL / audio.stem
    if not all((stem_dir / f"{s}.wav").exists() for s in STEMS):
        subprocess.run([py, "-m", "demucs", "-n", MODEL, "-d", device, "-o", str(out), str(audio)],
                       check=True)
    return stem_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--demucs-python", default=os.path.expanduser("~/dev/ai/demucs/.venv/bin/python"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    run = a.run.expanduser().resolve()
    audio, an = run / "final" / "audio.wav", run / "analysis"

    stem_dir = separate(audio, an / "stems", a.demucs_python, a.device)
    mix, sr = mono(audio)
    mix_db = db(np.sqrt((mix ** 2).mean()))

    entries, table = [], []
    for stem, (sid, label) in STEMS.items():
        x, ssr = mono(stem_dir / f"{stem}.wav")  # demucs writes 44.1 kHz; the mix is 48 kHz
        level = round(float(db(np.sqrt((x ** 2).mean())) - mix_db), 1)
        at = onset(db(frame_rms(x, ssr)))
        bleed = level < BLEED_DB
        table.append((stem, level, at, bleed))
        if not bleed and at is not None:
            entries.append({"id": sid, "label": label, "at": at,
                            "by": f"{MODEL} {stem} stem, first sustained energy ({ENTRY_DB:g} dB re its p95, gaps <= {GAP:g} s)"})

    words = json.loads((an / "heard.json").read_text())["words"]
    entries.append({"id": "words", "label": "words", "at": round(words[0]["start"], 2),
                    "by": f"heard.json: first word recognised and aligned (\"{words[0]['word']}\")"})

    chord_lab = an / "machine_listening" / "sheetsage" / "state" / "step_32" / "chord.lab"
    for line in chord_lab.read_text().splitlines():
        start, _, chord = line.split("\t")
        if chord != "N":
            entries.append({"id": "chords", "label": "the chords", "at": round(float(start), 2),
                            "by": f"SheetSage chord.lab on the finished state: first chord ({chord})"})
            break

    entries.sort(key=lambda e: e["at"])
    (an / "entries.json").write_text(json.dumps({
        "by": f"{MODEL} stems of the finished take for instruments and voice (onset rule: "
              f"{ENTRY_DB:g} dB re stem p95 for {HOLD:g} s, gaps <= {GAP:g} s; stems >{-BLEED_DB:g} dB under the mix dropped "
              f"as bleed); heard.json for words; SheetSage for chords",
        "entries": entries}, indent=1) + "\n")

    print(f"{'stem':8} {'level vs mix':>12} {'onset':>7}  note")
    for stem, level, at, bleed in table:
        print(f"{stem:8} {level:>9.1f} dB {at if at is not None else '-':>7}  {'bleed, dropped' if bleed else ''}")
    print("\nentries:")
    for e in entries:
        print(f"  {e['id']:7} {e['at']:6.2f}  {e['by']}")


if __name__ == "__main__":
    main()
