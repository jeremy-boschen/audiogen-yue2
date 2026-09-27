"""Data for the microscope explorers: three WebGL pages over finished microscope runs.

`bin/explore.py` writes the pages; this module turns run directories into the
compact arrays they draw. Everything here reads finished runs and writes only
into the output directory it is given (plus a feature cache inside it).

The pages load their data as `<script src="data/*.js">` files that assign one
global each, because Chrome refuses `fetch()` of local JSON over file://.

What the numbers are: spectrogram envelopes, correlations and projections are
signal statistics of decoded audio. They locate where things change and which
takes resemble each other as signals; they are not listening results, and the
pages say so wherever they show one.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import microscope as scope
from . import score as score_module

COLUMNS = 600          # terrain time columns (a third of a second each for 200 s)
BANDS = 64             # terrain log-frequency bands, 40 Hz .. 16 kHz, as microscope.envelope
MAP_COLUMNS = 400      # take-map feature resolution
MAP_BANDS = 48         # microscope.envelope's default, the one envelope_similarity uses
MAP_SECONDS = 200.0
MIN_TAKE_SECONDS = 190.0
ASSETS = Path(__file__).with_name("explore_assets")


# --- small pure helpers ----------------------------------------------------------

def pool_columns(grid: np.ndarray, columns: int) -> np.ndarray:
    """Average consecutive rows of a [frames, bands] grid into `columns` rows."""
    grid = np.asarray(grid, dtype=np.float64)
    frames = len(grid)
    if frames < columns:
        index = np.minimum((np.arange(columns) * frames) // columns, frames - 1)
        return grid[index]
    edges = np.linspace(0, frames, columns + 1).astype(int)
    sums = np.add.reduceat(grid, edges[:-1], axis=0)
    return sums / np.diff(edges)[:, None]


def quantize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map [lo, hi] linearly onto 0..255, clipping outside."""
    scaled = (np.asarray(values, dtype=np.float64) - lo) / (hi - lo if hi != lo else 1.0)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def band_correlation(grid: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Per band (column of a [time, bands] grid), Pearson correlation over time with `final`.

    A band with no variation in either grid gets 0: nothing to agree about.
    """
    a = np.asarray(grid, dtype=np.float64)
    b = np.asarray(final, dtype=np.float64)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denominator = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    numerator = (a * b).sum(axis=0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def settled_from(series: np.ndarray, threshold: float) -> int | None:
    """First index from which every later value is >= threshold, else None."""
    series = np.asarray(series)
    for index in range(len(series)):
        if (series[index:] >= threshold).all():
            return index
    return None


def band_edges(bands: int, rate: int = 48000) -> list[float]:
    """The band edges microscope.envelope uses."""
    return [float(x) for x in np.geomspace(40, min(16000, rate / 2), bands + 1)]


def occupied_bands(bands: int, rate: int = 48000, frame: int = 4096) -> np.ndarray:
    """Indices of envelope bands that contain at least one FFT bin.

    microscope.envelope's lowest log bands are narrower than one bin at
    frame 4096, so envelope() drops them; these are the survivors' indices
    into the nominal bands, for labelling their edges.
    """
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    index = np.digitize(freqs, np.geomspace(40, min(16000, rate / 2), bands + 1)) - 1
    return np.array(sorted(set(int(i) for i in index if 0 <= i < bands)), dtype=int)


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def js_global(name: str, value) -> str:
    """A script that assigns `value` (JSON-serialisable) to window[name]."""
    return f"window.{name} = {json.dumps(value, separators=(',', ':'), allow_nan=False)};\n"


def normalized(vectors: np.ndarray) -> np.ndarray:
    """Centre each row and scale it to unit length (so dot products are correlations)."""
    v = np.asarray(vectors, dtype=np.float64)
    v = v - v.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 0)


def classical_mds(distances: np.ndarray, dims: int = 3) -> np.ndarray:
    """Classical (Torgerson) MDS: coordinates whose Euclidean distances approximate `distances`."""
    d = np.asarray(distances, dtype=np.float64)
    n = len(d)
    if n == 0:
        return np.zeros((0, dims))
    j = np.eye(n) - 1.0 / n
    b = -0.5 * j @ (d ** 2) @ j
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1][:dims]
    coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
    if coords.shape[1] < dims:
        coords = np.pad(coords, ((0, 0), (0, dims - coords.shape[1])))
    return coords


def project_takes(features: np.ndarray, power: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """3-D positions and the correlation matrix for a stack of take feature vectors.

    Distance is sqrt(2(1 - r)) on centred, unit-length features (r = envelope
    correlation), raised to `power` so near-duplicates (noise seeds, long ODE
    solves) do not collapse onto one point. Axes carry no meaning.
    """
    unit = normalized(features)
    corr = np.clip(unit @ unit.T, -1.0, 1.0)
    distance = np.sqrt(np.clip(2.0 * (1.0 - corr), 0, None)) ** power
    coords = classical_mds(distance, 3)
    scale = np.abs(coords).max() or 1.0
    return coords / scale, corr


# --- the score, in time ------------------------------------------------------------

def voice_timeline(events, seconds_per_unit: float) -> dict:
    """Sounding notes of one voice as onset/duration in score seconds; ties merged."""
    t0, dur, pitch, section = [], [], [], []
    clock = Fraction(0)
    for event in events:
        length = float(event.units) * seconds_per_unit
        start = float(clock) * seconds_per_unit
        clock += event.units
        if event.kind != "note":
            continue
        if not event.onset and dur:
            dur[-1] = round(dur[-1] + length, 4)
            continue
        t0.append(round(start, 4))
        dur.append(round(length, 4))
        pitch.append(event.pitch)
        section.append(event.section)
    return {"t0": t0, "dur": dur, "pitch": pitch, "section": section,
            "seconds": round(float(clock) * seconds_per_unit, 3)}


def section_spans(score: score_module.Score, seconds_per_unit: float) -> list[dict]:
    """Each score section's start and end in score time, from the vocal voice's clock."""
    vocal = score_module.vocal_voice(score) or next(iter(score.voices), None)
    if vocal is None:
        return []
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}
    clock = Fraction(0)
    for event in score.voices[vocal]:
        start = float(clock) * seconds_per_unit
        clock += event.units
        starts.setdefault(event.section, start)
        ends[event.section] = float(clock) * seconds_per_unit
    return [{"index": i, "label": label, "start": round(starts[i], 3), "end": round(ends[i], 3)}
            for i, label in enumerate(score.sections) if i in starts]


def score_layer(abc_text: str) -> dict:
    """Everything the stack draws from the ABC: voices as timed notes, sections, header."""
    score = score_module.parse(abc_text)
    spu = score.seconds_per_unit() or 0.125
    voices = {name: voice_timeline(events, spu) for name, events in score.voices.items()}
    return {"tempo": score.headers.get("Q"), "key": score.headers.get("K"),
            "meter": score.headers.get("M"), "unit": score.headers.get("L"),
            "seconds_per_unit": spu, "voices": voices, "sections": section_spans(score, spu),
            "seconds": max((v["seconds"] for v in voices.values()), default=0.0),
            "pitch_note": "pitches as score.parse reads them: the key signature is not applied"}


def plan_reveal(prefixes: dict[int, str], final_voices: dict) -> list[dict]:
    """At each plan checkpoint, how many notes of each voice the prefix had written.

    Generation is autoregressive, so a prefix's notes are the first notes of the
    final plan; the count is enough to reveal them in order. A note cut in half
    by the prefix end counts as written.
    """
    out = []
    for tokens in sorted(prefixes):
        layer = score_layer(prefixes[tokens])
        counts = {name: min(len(layer["voices"].get(name, {"t0": []})["t0"]), len(v["t0"]))
                  for name, v in final_voices.items()}
        out.append({"tokens": tokens, "notes": counts,
                    "score_seconds": round(max((v["seconds"] for v in layer["voices"].values()),
                                               default=0.0), 3),
                    "sections": len([s for s in layer["sections"]])})
    return out


# --- audio -> grids, cached ----------------------------------------------------------

def _stat_key(path: Path, *params) -> str:
    stat = path.stat()
    text = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{params}"
    return hashlib.sha1(text.encode()).hexdigest()[:20]


def read_mono(path: Path, seconds: float | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf
    info = sf.info(str(path))
    frames = -1 if seconds is None else int(seconds * info.samplerate)
    audio, rate = sf.read(str(path), dtype="float32", frames=frames, always_2d=True)
    return audio.mean(axis=1), rate


def envelope_grid(path: Path, columns: int, bands: int, seconds: float | None = None) -> np.ndarray:
    """microscope.envelope of the file, pooled to [columns, bands] (log10 power)."""
    mono, rate = read_mono(path, seconds)
    return pool_columns(scope.envelope(mono, rate, bands=bands), columns)


def _grid_job(args):
    path, columns, bands, seconds, cache = args
    path, cache = Path(path), Path(cache) if cache else None
    target = cache / f"{_stat_key(path, columns, bands, seconds)}.npy" if cache else None
    if target is not None and target.exists():
        return np.load(target)
    grid = envelope_grid(path, columns, bands, seconds).astype(np.float32)
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, grid)
    return grid


def grids(paths: list[Path], columns: int, bands: int, cache: Path | None,
          seconds: float | None = None) -> list[np.ndarray]:
    return [_grid_job((str(p), columns, bands, seconds, str(cache) if cache else None)) for p in paths]


def waveform_peaks(path: Path, columns: int = 2000) -> np.ndarray:
    """[columns, 2] int8: min and max of the mono mix per column, scaled by the file's peak."""
    mono, _ = read_mono(path)
    edges = np.linspace(0, len(mono), columns + 1).astype(int)
    lo = np.minimum.reduceat(mono, edges[:-1])
    hi = np.maximum.reduceat(mono, edges[:-1])
    peak = float(np.abs(mono).max()) or 1.0
    return np.stack([np.rint(lo / peak * 127), np.rint(hi / peak * 127)], axis=1).astype(np.int8)


# --- run readers ---------------------------------------------------------------------

def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def is_finished(run: Path) -> bool:
    return bool((read_json(run / "metadata.json") or {}).get("finished"))


def rel(path: Path, out: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), Path(out).resolve())).as_posix()


def _metric_rows(rows: list[dict]) -> dict[int, dict]:
    return {row["step"]: row for row in rows or []}


def focus_data(run: Path, out: Path, cache: Path | None = None) -> dict:
    """Per-step spectrogram terrains (state and predicted final), settling, metrics, audio paths."""
    run = Path(run)
    metrics = read_json(run / "flow/chunk_000/metrics.json") or {}
    steps = metrics.get("steps") or len(sorted((run / "flow_audio_raw/chunk_000").glob("step_*.wav"))) - 1
    names = [f"step_{s:02d}.wav" for s in range(steps + 1)]
    state_paths = [run / "flow_audio_raw/chunk_000" / n for n in names]
    pred_paths = [run / "flow_predicted_audio_raw/chunk_000" / n for n in names]
    grid_list = grids(state_paths + pred_paths, COLUMNS, BANDS, cache)
    keep = occupied_bands(BANDS)
    # microscope.envelope already drops the bands with no FFT bin; `keep`
    # names which of the nominal bands survive, for their edges.
    state = np.stack(grid_list[:steps + 1])
    predicted = np.stack(grid_list[steps + 1:])
    edges = np.array(band_edges(BANDS))
    final = state[-1]
    hi = float(np.percentile(np.concatenate([state.ravel(), predicted.ravel()]), 99.9))
    lo = max(float(np.percentile(final, 0.5)), hi - 7.0)   # 70 dB window at most

    def settle(stack):
        return np.stack([band_correlation(g, final) for g in stack])

    settle_state, settle_pred = settle(state), settle(predicted)
    envelopes = read_json(run / "analysis/envelopes.json") or {}
    bands = read_json(run / "analysis/bands.json") or {}
    lat = _metric_rows(metrics.get("latent"))
    plat = _metric_rows(metrics.get("predicted_latent"))
    aud = _metric_rows(metrics.get("audio"))
    paud = _metric_rows(metrics.get("predicted_audio"))
    band_state = bands.get("flow_audio_raw", {}).get("chunk_000", {})
    band_pred = bands.get("flow_predicted_audio_raw", {}).get("chunk_000", {})

    def pick(row, *keys):
        return {k: row.get(k) for k in keys} if row else {}

    per_step = []
    for s in range(steps + 1):
        key = f"step_{s:02d}"
        per_step.append({
            "step": s, "t": (metrics.get("t") or {}).get(str(s)),
            "state": {**pick(lat.get(s), "cosine_final", "delta_final_relative"),
                      **pick(aud.get(s), "correlation_final", "rms", "spectral_centroid_hz"),
                      "envelope": envelopes.get("flow_state", {}).get(key),
                      "bands": band_state.get(str(s))},
            "predicted": {**pick(plat.get(s), "cosine_final", "delta_final_relative"),
                          **pick(paud.get(s), "correlation_final", "rms", "spectral_centroid_hz"),
                          "envelope": envelopes.get("flow_predicted", {}).get(key),
                          "bands": band_pred.get(str(s))},
        })

    def layers(stack):   # [step, band, time] so each step is one texture layer, rows = bands
        return b64(quantize(np.transpose(stack, (0, 2, 1)), lo, hi))

    listening = [run / "flow_audio_listening/chunk_000" / n for n in names]
    pred_listening = [run / "flow_predicted_audio_listening/chunk_000" / n for n in names]
    timeline = read_json(run / "analysis/timeline.json") or {}
    abc = run / "plan/final.abc" if (run / "plan/final.abc").exists() else run / "take/score.abc"
    sections = score_layer(abc.read_text())["sections"] if abc.exists() else []
    return {
        "run": run.name if run.name not in ("on-cpu", "on") else f"{run.parent.name}/{run.name}",
        "seconds": float(metrics.get("end_frame", 5000) - metrics.get("start_frame", 0)) / scope.SEMANTIC_RATE,
        "steps": steps, "columns": COLUMNS, "bands": int(len(keep)),
        "band_lo_hz": [round(float(x), 2) for x in edges[keep]],
        "band_hi_hz": [round(float(x), 2) for x in edges[keep + 1]],
        "bands_note": f"{BANDS - len(keep)} of {BANDS} log bands hold no FFT bin at frame 4096 and are omitted",
        "range_log10": [lo, hi],
        "state": layers(state), "predicted": layers(predicted),
        "settle_state": b64(quantize(settle_state, 0.0, 1.0)),
        "settle_predicted": b64(quantize(settle_pred, 0.0, 1.0)),
        "settled_from": {
            "state": [settled_from(settle_state[:, b], 0.9) for b in range(len(keep))],
            "predicted": [settled_from(settle_pred[:, b], 0.9) for b in range(len(keep))]},
        "metrics": per_step, "sections": sections,
        "signal_thresholds": ((timeline.get("metrics") or {}).get("chunk_000")
                              if isinstance(timeline, dict) else None),
        "audio": {"state": [rel(p, out) for p in listening],
                  "predicted": [rel(p, out) for p in pred_listening],
                  "finished": rel(listening[-1], out)},
        "audio_note": "peak-normalised listening copies (-1 dBFS); 'finished' is step 32, which is the take",
        "listening": listening_marks(run),
        # Where the local server writes marks (bin/explore.py serve); publish clears it.
        "annotate": rel(run, out) if (run / "analysis/annotations.json").exists() else None,
    }


# What each listening question is called on the page: what a listener hears, in plain words.
LISTENING_LABELS = {
    "beat_recognizable": "The beat", "bass_recognizable": "The bass", "harmony_recognizable": "The chords",
    "vocal_present": "A voice", "vocal_melody_recognizable": "The melody", "words_partially_intelligible": "Some words",
    "words_intelligible": "The words", "instrument_identity_recognizable": "The instruments",
    "stereo_image_established": "Stereo width", "ambience_reverb_established": "The room (reverb)",
    "transients_sharp": "Crisp hits", "essentially_final": "Sounds finished",
}


def listening_marks(run: Path) -> dict:
    """The listening checklist as the page needs it: every answer given so far, per view and step.

    Only a listener fills this in (analysis/annotations.json, by hand or from the page); nothing
    here is measured. Unanswered questions are left out.
    """
    notes = read_json(Path(run) / "analysis/annotations.json") or {}
    marks = {}
    for view, key in (("state", "ode"), ("predicted", "ode_predicted")):
        steps = (notes.get(key) or {}).get("chunk_000") or {}
        marks[view] = {str(s): {f: v for f, v in (answers or {}).items() if v is not None}
                       for s, answers in steps.items() if any(v is not None for v in (answers or {}).values())}
    return {"fields": [[f, LISTENING_LABELS.get(f, f)] for f in scope.LISTENING], **marks}


LEAD = 0.3
PICKUP_SECONDS = 1.0


def fold_pickups(phrases: list[dict]) -> list[dict]:
    """A short phrase ending a section that runs straight into the next is that next phrase's pickup.

    The score writes the first word of a verse ("My" mother...) as a note at the end of the
    chorus before it; rest-gap phrasing makes that note a phrase of its own. It is folded into
    the phrase it leads into, which then starts on it. The note counts are the score's, left as
    they were.
    """
    phrases = sorted(phrases, key=lambda p: p["start"])
    out = []
    for i, phrase in enumerate(phrases):
        nxt = phrases[i + 1] if i + 1 < len(phrases) else None
        seconds = phrase.get("seconds") or 0
        if (nxt and seconds <= PICKUP_SECONDS and nxt["section_index"] != phrase["section_index"]
                and nxt["start"] - (phrase["start"] + seconds) < 0.05):
            nxt["seconds"] = round((nxt.get("seconds") or 0) + nxt["start"] - phrase["start"], 2)
            nxt["start"], nxt["pickup"] = phrase["start"], True
            continue
        out.append(phrase)
    return out


def stack_data(run: Path, out: Path, cache: Path | None = None) -> dict:
    """Score, semantic tokens, acoustic latent and audio of one take, on one time axis."""
    run = Path(run)
    abc_path = run / "plan/final.abc" if (run / "plan/final.abc").exists() else run / "take/score.abc"
    layer = score_layer(abc_path.read_text())
    alignment = read_json(run / "analysis/alignment.json") or {}
    phrases = []
    for section in alignment.get("sections", []):
        for number, phrase in enumerate(section.get("phrases") or []):
            if phrase.get("start_seconds") is None:
                continue
            phrases.append({"section": section.get("label"), "section_index": section.get("index"),
                            "number": number + 1, "start": phrase["start_seconds"],
                            "seconds": phrase.get("seconds"), "lyrics": phrase.get("lyrics"),
                            "syllables": phrase.get("syllables"), "melody_notes": phrase.get("melody_notes"),
                            "notes_per_syllable": phrase.get("notes_per_syllable"), "bin": phrase.get("bin")})
    phrases = fold_pickups(phrases)
    heard = read_json(run / "analysis/heard.json")
    if heard:
        # Each phrase gets the words heard from its start until the next phrase starts. A word that
        # starts up to LEAD seconds before a phrase belongs to it: singers land a little ahead of the score.
        bounds = [p["start"] for p in phrases[1:]] + [float("inf")]
        for i, (phrase, end) in enumerate(zip(phrases, bounds)):
            lo = phrase["start"] if i else float("-inf")
            phrase["heard"] = " ".join(w["word"] for w in heard["words"] if lo <= w["start"] + LEAD < end) or None
    tokens = np.load(run / "take/semantic.npy").ravel().astype(np.int64)
    latent = np.load(run / "final/latent.npy") if (run / "final/latent.npy").exists() else np.load(run / "take/latent.npy")
    latent = latent.reshape(-1, latent.shape[-1])
    lat_lim = float(np.percentile(np.abs(latent), 99.5)) or 1.0
    audio_path = run / "final/audio.wav"
    keep = occupied_bands(BANDS)
    edges = np.array(band_edges(BANDS))
    spectrogram = grids([audio_path], COLUMNS, BANDS, cache)[0]
    s_hi = float(np.percentile(spectrogram, 99.9))
    s_lo = max(float(np.percentile(spectrogram, 0.5)), s_hi - 9.0)

    prefixes = {}
    for raw in sorted((run / "plan/abc").glob("*.raw.txt")):
        prefixes[int(raw.name.split(".")[0])] = raw.read_text()
    plan = read_json(run / "plan/analysis.json") or {}
    reveal = plan_reveal(prefixes, layer["voices"]) if prefixes else []
    clock = plan.get("seconds_at_checkpoint") or {}
    for row in reveal:
        row["generation_seconds"] = clock.get(str(row["tokens"]))
    semantic = read_json(run / "semantic/analysis.json") or {}
    sem_marks = [{"tokens": int(k), "generation_seconds": v.get("generation_seconds")}
                 for k, v in sorted((semantic.get("checkpoints") or {}).items(), key=lambda kv: int(kv[0]))]
    return {
        "run": run.name if run.name not in ("on-cpu", "on") else f"{run.parent.name}/{run.name}",
        "seconds": len(tokens) / scope.SEMANTIC_RATE, "score": layer, "phrases": phrases,
        "song": {k: (alignment.get("song") or {}).get(k) for k in ("notes_per_syllable", "syllables", "melody_notes", "bin")},
        "semantic": {"ids": b64(tokens.astype(np.uint16)), "count": int(len(tokens)),
                     "max": int(tokens.max()) if len(tokens) else 0, "rate": scope.SEMANTIC_RATE,
                     "distinct": int(len(np.unique(tokens)))},
        "latent": {"frames": int(latent.shape[0]), "channels": int(latent.shape[1]), "limit": lat_lim,
                   "values": b64(quantize(latent.T, -lat_lim, lat_lim))},
        "spectrogram": {"columns": COLUMNS, "bands": int(len(keep)),
                        "band_lo_hz": [round(float(x), 2) for x in edges[keep]],
                        "band_hi_hz": [round(float(x), 2) for x in edges[keep + 1]],
                        "values": b64(quantize(spectrogram.T, s_lo, s_hi))},
        "waveform": b64(waveform_peaks(audio_path)),
        "plan": reveal, "semantic_checkpoints": sem_marks,
        "heard": {"source": heard["source"]} if heard else None,
        "audio": rel(audio_path, out),
    }


# --- takes ----------------------------------------------------------------------------

STUDY = re.compile(r"\.(?P<kind>[a-z_]+)\.s(?P<seed>\d+)$")
GROUP_LABELS = {"seed": "seed study", "fixed_abc": "fixed ABC", "fixed_semantic": "fixed semantic"}


def discover_runs(roots) -> list[Path]:
    """Finished run directories (metadata.json with "finished") under each root, sorted."""
    found = set()
    for root in roots:
        root = Path(root)
        candidates = [root] if (root / "metadata.json").exists() else []
        candidates += [m.parent for m in root.glob("*/metadata.json")]
        candidates += [m.parent for m in root.glob("*/*/metadata.json")]
        for run in candidates:
            if "00_explore" in run.parts:
                continue
            if is_finished(run):
                found.add(run.resolve())
    return sorted(found)


def _seed_of(run: Path) -> int | None:
    match = STUDY.search(run.name) or re.search(r"-s(?P<seed>\d+)", str(run))
    return int(match.group("seed")) if match else None


def takes_in_run(run: Path) -> list[dict]:
    """Every whole take a run holds: its own, plus noise-seed and N-step renders of its tokens."""
    run = Path(run)
    audio = run / "final/audio.wav" if (run / "final/audio.wav").exists() else run / "take/audio.flac"
    if not audio.exists():
        return []
    match = STUDY.search(run.name)
    group = GROUP_LABELS.get(match.group("kind"), match.group("kind")) if match else "baseline"
    hashes = read_json(run / "baseline/hashes.json") or {}
    alignment = read_json(run / "analysis/alignment.json") or {}
    nps = (alignment.get("song") or {}).get("notes_per_syllable")
    seed = _seed_of(run)
    label_run = run.name if run.name not in ("on-cpu", "off") else f"{run.parent.name}/{run.name}"
    takes = [{"label": f"seed {seed}" if seed is not None else label_run, "group": group, "seed": seed,
              "run": label_run, "path": audio, "pcm_hash": (hashes.get("pcm") or {}).get("hash"),
              "notes_per_syllable": nps, "detail": "the take itself"}]
    noise = read_json(run / "noise_seeds/analysis.json") or {}
    for row in noise.get("rows", []):
        path = run / f"noise_seeds/noise_s{row['noise_seed']}.wav"
        if path.exists():
            takes.append({"label": f"seed {seed} tokens, noise {row['noise_seed']}", "group": "noise seed",
                          "seed": seed, "noise_seed": row["noise_seed"], "run": label_run, "path": path,
                          "pcm_hash": row.get("pcm_hash"), "notes_per_syllable": nps,
                          "detail": "same plan and tokens, different acoustic noise"})
    ode = read_json(run / "ode_steps/analysis.json") or {}
    for row in ode.get("rows", []):
        path = run / f"ode_steps/steps_{row['ode_steps']:03d}.wav"
        if path.exists():
            takes.append({"label": f"{row['ode_steps']}-step solve", "group": "ODE steps", "seed": seed,
                          "ode_steps": row["ode_steps"], "run": label_run, "path": path,
                          "pcm_hash": row.get("pcm_hash"), "notes_per_syllable": nps,
                          "detail": "same tokens and noise, solver run for this many steps"})
    return takes


def dedupe_takes(takes: list[dict]) -> list[dict]:
    """Merge takes with the same PCM hash (an --off twin, a study repeating the baseline)."""
    kept, by_hash = [], {}
    for take in takes:
        h = take.get("pcm_hash")
        if h and h in by_hash:
            by_hash[h].setdefault("aliases", []).append(f"{take['run']} ({take['group']})")
            continue
        if h:
            by_hash[h] = take
        kept.append(take)
    return kept


def audio_seconds(path: Path) -> float:
    import soundfile as sf
    return sf.info(str(path)).duration


def map_data(roots, out: Path, cache: Path | None = None, log=print) -> dict:
    """All takes under `roots`, placed by spectrogram-envelope similarity over the first 200 s."""
    runs = discover_runs(roots)
    order = {"baseline": 0}
    # the observed run represents a PCM hash its --off twin shares: it has final/, noise_seeds/, ode_steps/
    runs.sort(key=lambda r: (order.get("baseline" if not STUDY.search(r.name) else "study", 1), r.name == "off", str(r)))
    takes, skipped = [], []
    for run in runs:
        for take in takes_in_run(run):
            seconds = audio_seconds(take["path"])
            if seconds < MIN_TAKE_SECONDS:
                skipped.append(f"{take['run']}: {take['label']} is {seconds:.0f} s")
                continue
            takes.append(take)
    takes = dedupe_takes(takes)
    for line in skipped:
        log(f"map: skipped (too short to compare over {MAP_SECONDS:.0f} s) {line}")
    if not takes:
        return {"takes": [], "skipped": skipped}
    features = grids([t["path"] for t in takes], MAP_COLUMNS, MAP_BANDS, cache, MAP_SECONDS)
    keep = occupied_bands(MAP_BANDS)
    matrix = np.stack([f.ravel() for f in features])
    coords, corr = project_takes(matrix)
    base = next((i for i, t in enumerate(takes) if t["group"] == "baseline"), 0)
    rows = []
    for i, take in enumerate(takes):
        row = {k: v for k, v in take.items() if k != "path"}
        row["audio"] = rel(take["path"], out)
        row["pos"] = [round(float(x), 4) for x in coords[i]]
        row["similarity_to_first_baseline"] = round(float(corr[i, base]), 4)
        rows.append(row)
    return {"takes": rows, "similarity": np.round(corr, 4).tolist(), "skipped": skipped,
            "runs": [str(r) for r in runs],
            "method": ("each take: microscope.envelope (48 log bands, 40 Hz-16 kHz, phase discarded) over the "
                       f"first {MAP_SECONDS:.0f} s, pooled to {MAP_COLUMNS} columns, empty bands dropped; similarity = correlation of "
                       "those envelopes; positions = classical MDS of sqrt(2(1-r))^0.3 (the power spreads near-duplicates apart)")}


# --- writing ---------------------------------------------------------------------------

def install_assets(out: Path) -> None:
    """Copy the page shells, app scripts, CSS and vendored three.js into `out`."""
    out = Path(out)
    for name in ("app", "vendor"):
        source = ASSETS / name
        if source.exists():
            shutil.copytree(source, out / name, dirs_exist_ok=True)
    for page in ASSETS.glob("*.html"):
        shutil.copy2(page, out / page.name)
    shutil.copy2(ASSETS / "explore.css", out / "explore.css")


def write_data(out: Path, name: str, global_name: str, value: dict) -> Path:
    target = Path(out) / "data" / f"{name}.js"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(js_global(global_name, value))
    return target
