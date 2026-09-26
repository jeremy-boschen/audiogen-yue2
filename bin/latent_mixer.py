#!/usr/bin/env python3
"""A live mixer over a finished take's 64 latent channels.

    bin/latent_mixer.py RUN [--port 8770]

Loads RUN's final latent and its decoder, then serves a page on localhost that
loops an excerpt of the take. Every change to a channel re-decodes the excerpt
(~0.5 s for 10 s of audio on this Mac) and the loop crossfades to it in place.
Nothing is written unless you press Snapshot, which saves the edited excerpt
and its settings under RUN/latent_mixer/.
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "bin"))

from audiogen import latent_mixer as mixer  # noqa: E402

RATE = 48000
PAGE = HERE / "src" / "audiogen" / "latent_mixer.html"


def wav_bytes(audio: np.ndarray) -> bytes:
    import soundfile
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, RATE, subtype="FLOAT", format="WAV")
    return buffer.getvalue()


class Mixer:
    def __init__(self, run: pathlib.Path, models: pathlib.Path, quiet: bool):
        import microscope
        self.run = run
        self.latent = np.load(run / "final" / "latent.npy").astype(np.float32)
        self.whole = mixer.stats(self.latent)
        song, _ = microscope.song_for_run(run, str(models))
        self.pipe = microscope.renderer.build_pipeline(models, song, progress=not quiet)
        self.lock = threading.Lock()          # one decode at a time on the GPU
        found = next((d / "00_latent_directions" / "directions.npz" for d in run.parents
                      if (d / "00_latent_directions" / "directions.npz").exists()), None)
        self.basis = dict(np.load(found)) if found else None

    def excerpt(self, start: float, seconds: float) -> slice:
        a = max(0, min(len(self.latent) - 25, round(start * 25)))
        return slice(a, min(len(self.latent), a + max(25, round(seconds * 25))))

    def decode(self, body: dict) -> tuple[np.ndarray, dict]:
        span = self.excerpt(float(body.get("start", 0)), float(body.get("seconds", 10)))
        part = self.latent[span]
        # A held tone: one frame (or the take's average frame) repeated for the whole loop,
        # so an edit changes one steady sound instead of a moving mix.
        freeze = body.get("freeze")
        if freeze == "moment":
            at = max(0, min(len(self.latent) - 1, round(float(body.get("moment", body.get("start", 0))) * 25)))
            part = np.repeat(self.latent[at:at + 1], span.stop - span.start, axis=0)
        elif freeze == "average":
            part = np.repeat(self.whole["mean"][None, :].astype(np.float32), span.stop - span.start, axis=0)
        edited = part if body.get("bypass") else mixer.apply(part, body.get("edits") or mixer.neutral(), self.whole, self.basis)
        with self.lock:
            clock = time.perf_counter()
            audio = np.asarray(self.pipe.decode(edited), dtype=np.float32)
            took = time.perf_counter() - clock
        return audio, {"decode_seconds": round(took, 3), "start": span.start / 25, "seconds": (span.stop - span.start) / 25,
                       **mixer.signal_stats(audio, RATE)}


def handler_for(state: Mixer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body: bytes, kind: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def body(self) -> dict:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/info":
                info = {"run": state.run.name if state.run.name != "on-cpu" else f"{state.run.parent.name}/{state.run.name}",
                        "seconds": len(state.latent) / 25, "channels": mixer.CHANNELS,
                        "mean": state.whole["mean"].round(4).tolist(), "std": state.whole["std"].round(4).tolist(),
                        "directions": 0 if state.basis is None else 12,
                        "direction_variance": [] if state.basis is None else state.basis["variance"][:12].round(4).tolist(),
                        "direction_top": [] if state.basis is None else [
                            [[int(c), round(float(state.basis["components"][k][c]), 2)] for c in np.argsort(-np.abs(state.basis["components"][k]))[:3]]
                            for k in range(12)]}
                return self.send(200, json.dumps(info).encode(), "application/json")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                if self.path == "/api/decode":
                    audio, stats = state.decode(self.body())
                    return self.send(200, wav_bytes(audio), "audio/wav", {"X-Stats": json.dumps(stats)})
                if self.path == "/api/snapshot":
                    body = self.body()
                    audio, stats = state.decode(body)
                    folder = state.run / "latent_mixer"
                    name = time.strftime("%Y%m%d-%H%M%S")
                    from audiogen import microscope as scope
                    scope.write_wav(folder / f"{name}.wav", audio, RATE)
                    (folder / f"{name}.json").write_text(json.dumps({**body, "stats": stats}, indent=2) + "\n")
                    return self.send(200, json.dumps({"saved": str(folder / name)}).encode(), "application/json")
            except (ValueError, KeyError, TypeError) as exc:
                return self.send(400, str(exc).encode(), "text/plain")
            self.send(404, b"not found", "text/plain")
    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--models", default=str(HERE / "models"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    state = Mixer(pathlib.Path(args.run).expanduser().resolve(), pathlib.Path(args.models), args.quiet)
    state.decode({"start": 0, "seconds": 2})                 # load the decoder before the first request
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(state))
    print(f"latent mixer on http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
