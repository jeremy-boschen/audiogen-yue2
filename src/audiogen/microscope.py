"""Watch a take being made: ABC tokens, semantic tokens, and every acoustic ODE state.

The engine exposes three observation points and this module is their only
consumer: `on_token` for both sampled stages, and `on_step` (a
`yue2.nar.FlowStep`) for the acoustic solve. Observation must not change what
is generated, so the rules are strict:

* callbacks only append -- no decoding, no file I/O, no device compute;
* an ODE state is copied to host memory as it is reported (`capture="cpu"`),
  or held by reference (`capture="reference"`), which the engine permits
  because it never writes a state after reporting it. Which of the two leaves
  Metal undisturbed is measured, not assumed: run `bin/microscope.py verify`;
* everything is written after the take is finished.

What a token checkpoint is: generation is autoregressive, so tokens once
emitted are never revised and the prefix at N is exactly the first N tokens of
the final stream. The callbacks therefore add arrival times and a check that
the observed stream is the returned one, not information that a slice of the
final stream would lack. The ODE is different: its intermediate states exist
nowhere else.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hashes, score as score_module

PLAN_CHECKPOINTS = (50, 100, 200, 400, 800, 1600, 3200)
SEMANTIC_CHECKPOINTS = (250, 500, 1000, 2000, 4000, 8000)
SEMANTIC_RATE = 25                     # tokens (and latent frames) per second

# The manual listening checklist, per ODE step. Values are true/false/null.
LISTENING = ("beat_recognizable", "bass_recognizable", "harmony_recognizable", "vocal_present",
             "vocal_melody_recognizable", "words_partially_intelligible", "words_intelligible",
             "instrument_identity_recognizable", "stereo_image_established",
             "ambience_reverb_established", "transients_sharp", "essentially_final")


def checkpoints(total: int, marks) -> list[int]:
    """The marks that fall inside `total`, plus `total` itself."""
    return sorted({m for m in marks if 0 < m < total} | ({total} if total else set()))


def select_steps(spec, steps: int) -> set[int]:
    """'all' or an iterable of step numbers, 0 (the noise) through `steps`."""
    if spec in (None, "all"):
        return set(range(steps + 1))
    return {int(s) for s in spec if 0 <= int(s) <= steps}


@dataclass
class Recorder:
    """The callbacks handed to the engine, and what they collected."""
    ode_checkpoints: object = "all"
    capture: str = "cpu"
    tokens: dict = field(default_factory=lambda: {"abc": [], "semantic": []})
    arrivals: dict = field(default_factory=lambda: {"abc": [], "semantic": []})
    flow: dict = field(default_factory=dict)       # chunk index -> {"meta": ..., "states": {step: tensor}}
    started: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        if self.capture not in ("cpu", "reference"):
            raise ValueError("capture must be 'cpu' or 'reference'")

    def on_token(self, phase, token):
        self.tokens.setdefault(phase, []).append(token)
        self.arrivals.setdefault(phase, []).append(time.perf_counter() - self.started)

    def on_step(self, flow):
        if flow.step not in select_steps(self.ode_checkpoints, flow.steps):
            return
        chunk = self.flow.setdefault(flow.chunk_index, {
            "meta": {"chunk_index": flow.chunk_index, "chunk_count": flow.chunk_count,
                     "start_frame": flow.start, "end_frame": flow.end, "lead_frames": flow.lead,
                     "steps": flow.steps, "dtype": str(flow.state.dtype).replace("torch.", ""),
                     "device": str(flow.state.device)},
            "states": {}})
        keep = (lambda x: x.detach().to("cpu", copy=True)) if self.capture == "cpu" else (lambda x: x.detach())
        chunk["states"][flow.step] = keep(flow.state)
        chunk.setdefault("t", {})[flow.step] = flow.t
        if flow.velocity is not None:
            chunk.setdefault("velocity", {})[flow.step] = keep(flow.velocity)

    def stream(self, phase: str, end_token: int | None = None) -> list[int]:
        """The observed output of one stage, without its end token."""
        ids = list(self.tokens.get(phase, []))
        if ids and end_token is not None and ids[-1] == end_token:
            ids.pop()
        return ids

    def states(self, chunk_index: int) -> dict[int, np.ndarray]:
        """A chunk's captured states as FP32 host arrays (a widening, so exact)."""
        return {step: state.float().cpu().numpy()
                for step, state in sorted(self.flow[chunk_index]["states"].items())}

    def predicted(self, chunk_index: int) -> dict[int, np.ndarray]:
        """The solver's estimate of the solution at each state: x - t*v.

        Derived here in FP32 on the host from the reported state and velocity,
        so it is an observation of the solve, not a value the solver computed.
        The solution itself is its own estimate.
        """
        chunk = self.flow[chunk_index]
        velocity, t = chunk.get("velocity", {}), chunk["t"]
        out = {}
        for step, state in sorted(chunk["states"].items()):
            x = state.float().cpu().numpy()
            out[step] = x - np.float32(t[step]) * velocity[step].float().cpu().numpy() if step in velocity else x
        return out


# --- metrics -------------------------------------------------------------------

def latent_metrics(states: dict[int, np.ndarray]) -> list[dict]:
    final = states[max(states)]
    final_norm = float(np.linalg.norm(final)) or 1.0
    rows, previous = [], None
    for step, x in states.items():
        x64 = x.astype(np.float64)
        rows.append({
            "step": step, "shape": list(x.shape), "dtype": str(x.dtype),
            "min": float(x64.min()), "max": float(x64.max()), "mean": float(x64.mean()),
            "std": float(x64.std()), "norm": float(np.linalg.norm(x64)),
            "delta_previous": None if previous is None else float(np.linalg.norm(x64 - previous)),
            "delta_final": float(np.linalg.norm(x64 - final)),
            "delta_final_relative": float(np.linalg.norm(x64 - final)) / final_norm,
            "cosine_final": float((x64 * final).sum() / ((np.linalg.norm(x64) * final_norm) or 1.0)),
        })
        previous = x64
    return rows


def audio_metrics(audio: np.ndarray, final: np.ndarray | None, rate: int, frame: int = 4096) -> dict:
    """Cheap signal descriptors. They locate transitions; they are not what anything sounds like."""
    a = np.asarray(audio, dtype=np.float64)
    mono = a.mean(axis=1) if a.ndim == 2 else a
    hop = frame // 2
    count = max(0, (len(mono) - frame) // hop + 1)
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    centroids, widths, weights = [], [], []
    for start in range(0, count * hop, hop * 256):
        block = np.lib.stride_tricks.sliding_window_view(mono[start:start + hop * 255 + frame], frame)[::hop]
        spectrum = np.abs(np.fft.rfft(block * window, axis=1))
        total = spectrum.sum(axis=1)
        live = total > 1e-9
        if not live.any():
            continue
        s, t = spectrum[live], total[live]
        c = (s * freqs).sum(axis=1) / t
        centroids.append(c)
        widths.append(np.sqrt((s * (freqs - c[:, None]) ** 2).sum(axis=1) / t))
        weights.append(t)
    row = {
        "rms": float(np.sqrt((a ** 2).mean())),
        "peak": float(np.abs(a).max()),
        "zero_crossing_rate": float((np.diff(np.signbit(mono)) != 0).mean()) if len(mono) > 1 else 0.0,
        "spectral_centroid_hz": None, "spectral_bandwidth_hz": None,
    }
    if centroids:
        w = np.concatenate(weights)
        row["spectral_centroid_hz"] = float((np.concatenate(centroids) * w).sum() / w.sum())
        row["spectral_bandwidth_hz"] = float((np.concatenate(widths) * w).sum() / w.sum())
    if final is not None and np.shape(final) == np.shape(a):
        f = np.asarray(final, dtype=np.float64)
        da, df = a - a.mean(), f - f.mean()
        denominator = np.sqrt((da ** 2).sum() * (df ** 2).sum())
        row["correlation_final"] = float((da * df).sum() / denominator) if denominator else None
        row["difference_rms_final"] = float(np.sqrt(((a - f) ** 2).mean()))
    return row


# --- writing a run -------------------------------------------------------------

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_wav(path: Path, audio: np.ndarray, rate: int, subtype: str = "FLOAT") -> None:
    import soundfile
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, audio, rate, subtype=subtype)


def listening_copy(audio: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    """Peak-normalised for comfortable listening. Never the authoritative file."""
    peak = float(np.abs(audio).max())
    return audio if peak == 0 else audio * (10 ** (peak_dbfs / 20) / peak)


def write_tokens(directory: Path, name: str, ids: list[int], marks: list[int], decode=None) -> dict:
    """Prefix checkpoints of one token stream: ids, and raw decoded text when a decoder is given."""
    out = {}
    (directory / "tokens").mkdir(parents=True, exist_ok=True)
    for mark in marks:
        prefix = ids[:mark]
        np.save(directory / "tokens" / f"{mark:06d}.npy", np.asarray(prefix, dtype=np.int32))
        entry = {"tokens": len(prefix), "hash": hashes.hash_tokens(prefix)}
        if decode is not None:
            text = decode(prefix)
            (directory / name).mkdir(parents=True, exist_ok=True)
            (directory / name / f"{mark:06d}.raw.txt").write_text(text)
            valid = score_module.valid_prefix(text)
            (directory / name / f"{mark:06d}.valid.abc").write_text(valid)
            entry["chars"] = len(text)
        out[mark] = entry
    return out


def write_capture(run: Path, pipe, take, recorder: Recorder, *, rate: int) -> dict:
    """Serialise tokens, ABC prefixes and ODE states. Called after the take is done."""
    from yue2.protocol import CODEC_OFFSET

    report = {"integrity": {}}
    plan_ids = list(take.result.semantic.plan.abc_ids)
    observed_abc = recorder.tokens.get("abc", [])
    # The end token is reported to on_token but is not part of abc_ids.
    report["integrity"]["abc_stream_matches"] = observed_abc[:len(plan_ids)] == plan_ids
    report["integrity"]["abc_observed"] = len(observed_abc)
    semantic = list(take.semantic)
    observed_semantic = [t - CODEC_OFFSET for t in recorder.tokens.get("semantic", [])]
    carried = take.carried.get("tokens", 0) if take.carried else 0   # reproduced, not sampled
    report["integrity"]["semantic_stream_matches"] = (
        observed_semantic[:len(semantic) - carried] == semantic[carried:])
    report["integrity"]["semantic_observed"] = len(observed_semantic)

    # Plan checkpoints: raw decoded prefixes are authoritative, valid prefixes derived.
    plan_dir = run / "plan"
    if plan_ids:
        marks = checkpoints(len(plan_ids), PLAN_CHECKPOINTS)
        report["plan"] = write_tokens(plan_dir, "abc", plan_ids, marks,
                                      decode=lambda ids: pipe.tokenizer.decode(ids))
        (plan_dir / "final.abc").write_text(take.score or "")
        prefixes = {m: (plan_dir / "abc" / f"{m:06d}.raw.txt").read_text() for m in marks}
        analysis = score_module.development(prefixes)
        arrivals = recorder.arrivals.get("abc", [])
        analysis["seconds_at_checkpoint"] = {m: round(arrivals[m - 1], 2) for m in marks if m <= len(arrivals)}
        write_json(plan_dir / "analysis.json", analysis)
    else:
        report["plan"] = {"note": "no generated plan: the score was supplied or CoT is off"}

    # Semantic checkpoints, codec-relative (CODEC_OFFSET removed) like SemanticResult.tokens.
    sem_dir = run / "semantic"
    marks = checkpoints(len(semantic), SEMANTIC_CHECKPOINTS)
    report["semantic"] = write_tokens(sem_dir, "semantic", semantic, marks)
    arrivals = recorder.arrivals.get("semantic", [])
    write_json(sem_dir / "analysis.json", {
        "ids": "codec-relative (CODEC_OFFSET removed), as SemanticResult.tokens",
        "codec_offset": CODEC_OFFSET, "rate": SEMANTIC_RATE, "carried_tokens": carried,
        "checkpoints": {m: {**report["semantic"][m], "seconds_of_music": round(m / SEMANTIC_RATE, 2),
                            "generation_seconds": round(arrivals[m - carried - 1], 2)
                            if 0 < m - carried <= len(arrivals) else None} for m in marks},
    })

    # ODE states, one directory per chunk so steps of different chunks never collide.
    report["flow"] = {}
    for index, chunk in sorted(recorder.flow.items()):
        directory = run / "flow" / f"chunk_{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        states = recorder.states(index)
        width = max(2, len(str(chunk["meta"]["steps"])))
        predicted = recorder.predicted(index)
        for step, state in states.items():
            np.save(directory / f"step_{step:0{width}d}.npy", state)
            np.save(directory / f"predicted_{step:0{width}d}.npy", predicted[step])
        write_json(directory / "metrics.json", {
            **chunk["meta"], "stored_dtype": "float32", "steps_captured": sorted(states),
            "t": {step: chunk["t"][step] for step in sorted(states)},
            "latent": latent_metrics(states), "predicted_latent": latent_metrics(predicted)})
        report["flow"][index] = {**chunk["meta"], "steps_captured": len(states)}
    return report


def decode_flow(run: Path, pipe, recorder: Recorder, final_audio: np.ndarray, *, rate: int,
                listening: bool = True) -> dict:
    """Decode every captured state through the pipeline's own VAE path.

    A single-chunk song decodes each state as the whole song. A chunk of a
    multi-chunk song is decoded alone, which is not how the song's own decode
    sees its edges; its audio is therefore compared with the final only where
    the lengths agree.
    """
    out = {}
    single = len(recorder.flow) == 1
    for index in sorted(recorder.flow):
        width = max(2, len(str(recorder.flow[index]["meta"]["steps"])))
        metrics = run / "flow" / f"chunk_{index:03d}" / "metrics.json"
        record = json.loads(metrics.read_text())
        for kind, states, raw_dir, listen_dir in (
                ("audio", recorder.states(index), "flow_audio_raw", "flow_audio_listening"),
                ("predicted_audio", recorder.predicted(index), "flow_predicted_audio_raw",
                 "flow_predicted_audio_listening")):
            rows = []
            for step, state in states.items():
                audio = pipe.decode(state)
                name = f"chunk_{index:03d}/step_{step:0{width}d}.wav"
                write_wav(run / raw_dir / name, audio, rate)
                if listening:
                    write_wav(run / listen_dir / name, listening_copy(audio), rate, "PCM_16")
                rows.append({"step": step, "pcm_hash": hashes.hash_array(audio), "frames": len(audio),
                             **audio_metrics(audio, final_audio if single else None, rate)})
            record[kind] = rows
        write_json(metrics, record)
        out[index] = record["audio"]
    return out


def annotation_template(recorder: Recorder) -> dict:
    return {
        "how": ("Set each field to true or false after listening to "
                "flow_audio_listening/chunk_NNN/step_SS.wav (the ODE state itself). "
                "flow_predicted_audio_listening/ holds the solver's estimate of the finished "
                "audio at the same step; annotate it under 'ode_predicted'. Leave null for not judged. "
                "'phrases' holds intelligibility per phrase from alignment.txt."),
        "ode": {f"chunk_{i:03d}": {str(step): {key: None for key in LISTENING}
                                   for step in sorted(chunk["states"])}
                for i, chunk in sorted(recorder.flow.items())},
        "ode_predicted": {f"chunk_{i:03d}": {str(step): {key: None for key in LISTENING}
                                             for step in sorted(chunk["states"])}
                          for i, chunk in sorted(recorder.flow.items())},
        "phrases": {},
        "notes": "",
    }


def timeline(run: Path) -> dict:
    """Combine annotations with metrics: the first step at which each judgment became true."""
    annotations = json.loads((run / "analysis" / "annotations.json").read_text())
    def firsts(steps):
        out = {}
        for key in LISTENING:
            judged = sorted((int(s), v[key]) for s, v in steps.items() if v.get(key) is not None)
            out[key] = next((s for s, v in judged if v), None) if judged else None
        return out

    result = {"ode": {}, "ode_predicted": {}, "metrics": {}}
    for chunk, steps in annotations.get("ode_predicted", {}).items():
        result["ode_predicted"][chunk] = firsts(steps)
    for chunk, steps in annotations.get("ode", {}).items():
        result["ode"][chunk] = firsts(steps)
        metrics_file = run / "flow" / chunk / "metrics.json"
        if metrics_file.exists():
            record = json.loads(metrics_file.read_text())
            corr = [(r["step"], r.get("correlation_final")) for r in record.get("audio", [])]
            result["metrics"][chunk] = {
                "correlation_final_above_0.9": next((s for s, c in corr if c is not None and c >= 0.9), None),
                "correlation_final_above_0.99": next((s for s, c in corr if c is not None and c >= 0.99), None),
                "latent_cosine_above_0.9": next((r["step"] for r in record["latent"] if r["cosine_final"] >= 0.9), None),
                "latent_cosine_above_0.99": next((r["step"] for r in record["latent"] if r["cosine_final"] >= 0.99), None),
                "predicted_correlation_final_above_0.9": next(
                    (r["step"] for r in record.get("predicted_audio", [])
                     if (r.get("correlation_final") or 0) >= 0.9), None),
                "predicted_correlation_final_above_0.99": next(
                    (r["step"] for r in record.get("predicted_audio", [])
                     if (r.get("correlation_final") or 0) >= 0.99), None),
            }
    return result


def timeline_markdown(run: Path, events: dict) -> str:
    lines = ["# Microscope timeline", "",
             "Metric thresholds locate change; they are not listening results. Listening rows",
             "come only from analysis/annotations.json and stay blank until someone listens.", ""]
    for chunk, firsts in events["ode"].items():
        guessed = events.get("ode_predicted", {}).get(chunk, {})
        lines += [f"## {chunk}", "", "| judgment | first step true: ODE state | first step true: predicted final |",
                  "|---|---|---|"]
        lines += [f"| {k.replace('_', ' ')} | {'' if v is None else v} | "
                  f"{'' if guessed.get(k) is None else guessed.get(k)} |" for k, v in firsts.items()]
        lines.append("")
        m = events["metrics"].get(chunk)
        if m:
            lines += ["| signal (not perceptual) | first step |", "|---|---|"]
            lines += [f"| {k.replace('_', ' ')} | {'' if v is None else v} |" for k, v in m.items()]
            lines.append("")
    return "\n".join(lines)


def metrics_csv(run: Path) -> str:
    rows = ["chunk,step,latent_norm,latent_delta_previous,latent_delta_final_relative,latent_cosine_final,"
            "rms,peak,spectral_centroid_hz,spectral_bandwidth_hz,zero_crossing_rate,correlation_final,"
            "difference_rms_final,predicted_latent_cosine_final,predicted_correlation_final,predicted_rms"]
    for metrics in sorted((run / "flow").glob("chunk_*/metrics.json")):
        record = json.loads(metrics.read_text())
        audio = {r["step"]: r for r in record.get("audio", [])}
        guess_audio = {r["step"]: r for r in record.get("predicted_audio", [])}
        guess = {r["step"]: r for r in record.get("predicted_latent", [])}
        for r in record["latent"]:
            a = audio.get(r["step"], {})
            g, ga = guess.get(r["step"], {}), guess_audio.get(r["step"], {})
            values = [record["chunk_index"], r["step"], r["norm"], r["delta_previous"],
                      r["delta_final_relative"], r["cosine_final"], a.get("rms"), a.get("peak"),
                      a.get("spectral_centroid_hz"), a.get("spectral_bandwidth_hz"),
                      a.get("zero_crossing_rate"), a.get("correlation_final"), a.get("difference_rms_final"),
                      g.get("cosine_final"), ga.get("correlation_final"), ga.get("rms")]
            rows.append(",".join("" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
                                 for v in values))
    return "\n".join(rows) + "\n"


BANDS = ((20, 150, "sub/bass"), (150, 600, "low-mid"), (600, 2500, "mid"), (2500, 8000, "presence"),
         (8000, 20000, "air"))


def band_correlations(audio: np.ndarray, final: np.ndarray, rate: int, bands=BANDS) -> dict:
    """Correlation with the final, per frequency band of the mono mix.

    Tests "coarse before fine" on the signal: if low bands settle at earlier
    steps than high ones, it shows here. A signal statistic, not a percept.
    """
    def spectrum(x):
        mono = np.asarray(x, dtype=np.float64)
        mono = mono.mean(axis=1) if mono.ndim == 2 else mono
        return np.fft.rfft(mono - mono.mean())
    a, f = spectrum(audio), spectrum(final)
    freqs = np.fft.rfftfreq((len(a) - 1) * 2, 1.0 / rate)
    out = {}
    for low, high, name in bands:
        sel = (freqs >= low) & (freqs < high)
        # Parseval: the band-limited time-domain correlation, computed in frequency.
        num = np.real((a[sel] * np.conj(f[sel])).sum())
        den = np.sqrt((np.abs(a[sel]) ** 2).sum() * (np.abs(f[sel]) ** 2).sum())
        out[name] = float(num / den) if den else None
    return out
