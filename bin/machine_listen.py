#!/usr/bin/env python
"""Where machine listeners first pick something out, step by step through the solve.

Runs under the SheetSage venv, which has SheetSage2's pinned transformers and mir_eval:
    ~/dev/ai/sheetsage/.venv/bin/python bin/machine_listen.py RUN [--audiocpp ~/dev/ai/audio.cpp]

For every step's audio in both views (flow_audio_listening = state, flow_predicted_audio_listening
= predicted), each listener transcribes what it can hear and compares that with its own transcription
of the finished take (state step 32):

  words        Qwen3-ASR (audio.cpp): share of the finished take's words heard, in order
  voice        Silero VAD (audio.cpp): seconds of voice, against the finished take's
  chords       SheetSage2: share of the song with the same major/minor chord (mir_eval.chord)
  beat         SheetSage2: beat F-measure, 70 ms window (mir_eval.beat)           scored, not marked
  melody       SheetSage2: vocal melody note F-measure, onset 100 ms, pitch 50 cents  scored, not marked

A listening field is marked at the first step from which its rule holds for every later step.
The rules were fixed before any per-step result was seen, and are not tuned to a listener's marks.
These are machines picking things out of a signal, not a person listening: the page shows them apart
from marks a listener made, and a machine can pick out what a person cannot yet hear, or the reverse.

Per-step results are kept under RUN/analysis/machine_listening/ so a re-run only does what is new.
Writes RUN/analysis/machine_marks.json.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
VIEWS = {"state": "flow_audio_listening", "predicted": "flow_predicted_audio_listening"}
ASR = ("qwen3_asr", "models/Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf")
VAD = ("silero_vad", "assets/framework/models/silero_vad")
# field: (score name, threshold, what the page says a machine did)
RULES = {
    "words_partially_intelligible": ("words", 0.2, "a speech recognizer picked out a fifth of the words"),
    "words_intelligible": ("words", 0.9, "a speech recognizer picked out nine in ten of the words"),
    "vocal_present": ("voice", 0.5, "a voice detector found at least half the finished take's voice"),
    "harmony_recognizable": ("chords", 0.8, "a transcriber read the same chords for 80% of the song"),
}
# Scored, but not made into marks. On Slow Down take 2 SheetSage2 does not agree with itself: state
# steps 26-31 sound all but identical to the finished take, yet its beats scored 0.66-0.79 and its
# melody 0.77-0.87 against its own reading of step 32, so an 80% rule marked step 31-32. That is the
# transcriber's repeatability, not the song forming. The thresholds were not lowered to fit.
UNMARKED = {"beat": "SheetSage2 beats agree with its own reading of the finished take only ~0.7 even on near-identical audio",
            "melody": "SheetSage2 melody agrees with its own reading of the finished take only ~0.8 even on near-identical audio"}
BY = {"words": "Qwen3-ASR 1.7B (audio.cpp)", "voice": "Silero VAD (audio.cpp)",
      "beat": "SheetSage2", "chords": "SheetSage2", "melody": "SheetSage2"}


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]+", " ", text.lower()).split()


def lab(path: pathlib.Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def cli(root: pathlib.Path, task: str, spec, audio: pathlib.Path, *extra) -> None:
    subprocess.run([str(root / "build/macos-metal-release/bin/audiocpp_cli"), "--task", task, "--family", spec[0],
                    "--model", str(root / spec[1]), "--backend", "metal", "--audio", str(audio), *extra],
                   check=True, capture_output=True, text=True)


def hear(root: pathlib.Path, wav: pathlib.Path, cache: dict) -> dict:
    """Recognized text and seconds of voice for one step's audio (cached by file)."""
    key = f"{wav.parent.parent.name}/{wav.name}"
    if key in cache:
        return cache[key]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        cli(root, "asr", ASR, wav, "--language", "en", "--text-out", str(tmp / "t.txt"))
        mono = tmp / "16k.wav"                        # Silero only takes 16 kHz
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "16000", str(mono)], check=True)
        cli(root, "vad", VAD, mono, "--segments-out", str(tmp / "v.json"))
        found = json.loads((tmp / "v.json").read_text()) if (tmp / "v.json").exists() else []   # none written: no voice
        found = found.get("segments", found) if isinstance(found, dict) else found
        cache[key] = {"text": (tmp / "t.txt").read_text().strip(),
                      "voice_seconds": round(sum(s["end_sample"] - s["start_sample"] for s in found) / 16000, 2)}
    return cache[key]


def compare(step_dir: pathlib.Path, ref_dir: pathlib.Path) -> dict:
    import mir_eval
    beat = lambda d: np.array([float(r[0]) for r in lab(d / "beat.lab")])
    out = {"beat": float(mir_eval.beat.f_measure(beat(ref_dir), beat(step_dir), f_measure_threshold=0.07))}
    ref_c, est_c = lab(ref_dir / "chord.lab"), lab(step_dir / "chord.lab")
    if ref_c and est_c:
        ri = np.array([[float(a), float(b)] for a, b, _ in ref_c]); ei = np.array([[float(a), float(b)] for a, b, _ in est_c])
        ei, el = mir_eval.util.adjust_intervals(ei, [c for *_, c in est_c], ri.min(), ri.max(),
                                                mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
        iv, rl, el = mir_eval.util.merge_labeled_intervals(ri, [c for *_, c in ref_c], ei, el)
        out["chords"] = float(mir_eval.chord.weighted_accuracy(mir_eval.chord.majmin(rl, el), mir_eval.util.intervals_to_durations(iv)))
    else:
        out["chords"] = 0.0
    notes = lambda d: lab(d / "melody_vocal.lab")
    rn, en = notes(ref_dir), notes(step_dir)
    if rn and en:
        hz = lambda ns: mir_eval.util.midi_to_hz(np.array([float(n[2]) for n in ns]))
        iv = lambda ns: np.array([[float(n[0]), float(n[1])] for n in ns])
        out["melody"] = float(mir_eval.transcription.precision_recall_f1_overlap(
            iv(rn), hz(rn), iv(en), hz(en), onset_tolerance=0.1, pitch_tolerance=50.0, offset_ratio=None)[2])
    else:
        out["melody"] = 0.0
    return out


def first_held(scores: list[float], threshold: float) -> int | None:
    """The first step from which every later step meets the threshold."""
    step = None
    for s in range(len(scores) - 1, -1, -1):
        if scores[s] < threshold:
            break
        step = s
    return step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--audiocpp", default=os.environ.get("AUDIOCPP", str(pathlib.Path.home() / "dev/ai/audio.cpp")))
    args = parser.parse_args()
    run, root = pathlib.Path(args.run).resolve(), pathlib.Path(args.audiocpp).expanduser()
    work = run / "analysis/machine_listening"
    work.mkdir(parents=True, exist_ok=True)
    cache_path = work / "heard.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    wavs = {view: sorted((run / folder / "chunk_000").glob("step_*.wav")) for view, folder in VIEWS.items()}
    reference = wavs["state"][-1]                     # state step 32 is the finished take
    for view, files in wavs.items():                  # SheetSage in its own venv, model loaded once per view
        links = work / "audio" / view
        links.mkdir(parents=True, exist_ok=True)
        for f in files:
            (links / f.name).unlink(missing_ok=True)
            (links / f.name).symlink_to(f)
        subprocess.run([sys.executable, str(HERE / "sheetsage_batch.py"), str(work / "sheetsage" / view),
                        *map(str, sorted(links.glob("*.wav")))], check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})

    ref_heard = hear(root, reference, cache)
    ref_words = words(ref_heard["text"])
    ref_dir = work / "sheetsage/state" / reference.stem
    out = {"listeners": {f: {"score": s, "threshold": t, "says": says, "by": BY[s]} for f, (s, t, says) in RULES.items()},
           "unmarked": UNMARKED}
    for view, files in wavs.items():
        scores = {k: [] for k in BY}
        for wav in files:
            heard = hear(root, wav, cache)
            cache_path.write_text(json.dumps(cache, indent=1))
            got = words(heard["text"])
            matched = sum(b.size for b in difflib.SequenceMatcher(None, ref_words, got, autojunk=False).get_matching_blocks())
            scores["words"].append(round(matched / max(1, len(ref_words)), 3))
            scores["voice"].append(round(min(1.0, heard["voice_seconds"] / max(1e-6, ref_heard["voice_seconds"])), 3))
            for k, v in compare(work / "sheetsage" / view / wav.stem, ref_dir).items():
                scores[k].append(round(v, 3))
            print(f"{view:9} {wav.stem}: " + "  ".join(f"{k} {v[-1]:.2f}" for k, v in scores.items()), flush=True)
        out[view] = {"scores": scores,
                     "first": {f: first_held(scores[s], t) for f, (s, t, _) in RULES.items()}}
    (run / "analysis/machine_marks.json").write_text(json.dumps(out, indent=1))
    for view in VIEWS:
        print(view, {f: s for f, s in out[view]["first"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
