#!/usr/bin/env python
"""The YuE2 generation microscope: watch one take being made, stage by stage.

    bin/microscope.py capture burn_it_down --run RUN          observe everything, write it all
    bin/microscope.py capture burn_it_down --run RUN --off    the same render, nothing observed
    bin/microscope.py compare RUN_A RUN_B                     stage hashes + decoded PCM, exact
    bin/microscope.py prefixes RUN --tokens 500,1000,2000     render semantic prefixes alone
    bin/microscope.py ode-steps RUN --steps 1,2,4,8,16,32     re-solve the fixed tokens with N steps
    bin/microscope.py annotate RUN --step 12 vocal_present=true words_intelligible=false
    bin/microscope.py annotate RUN --step 4 --predicted vocal_present=true
    bin/microscope.py annotate RUN --phrase verse.1.2 good
    bin/microscope.py timeline RUN                            annotations + metrics -> timeline.{json,md}
    bin/microscope.py study MANIFEST.json                     fixed-ABC / fixed-semantic / seed studies

A RUN is a directory. Runs belong in the studio (../audiogen/output/...), never
in this repo. Observation must not change the take: `compare` an `--off` run
with an observed one before trusting anything the observed one says.
"""
import argparse
import dataclasses
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from audiogen import hashes  # noqa: E402
from audiogen import microscope as scope  # noqa: E402
from audiogen import render as renderer  # noqa: E402
from audiogen import score as score_module  # noqa: E402
from audiogen import song as song_module  # noqa: E402

RATE = renderer.SAMPLE_RATE


def git_commit(path: pathlib.Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "describe", "--always", "--dirty", "--abbrev=40"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def engine_source() -> dict:
    import yue2
    from importlib.metadata import distribution
    source = pathlib.Path(yue2.__file__).resolve().parent
    direct = distribution("yue2-infer").read_text("direct_url.json")
    return {"path": str(source), "install": json.loads(direct) if direct else None,
            "commit": git_commit(source) if (source.parent.parent / ".git").exists() else None}


def machine() -> dict:
    import torch
    return {"macos": platform.mac_ver()[0], "machine": platform.machine(),
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True).stdout.strip(),
            "memory_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                               capture_output=True, text=True).stdout.strip() or 0),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available()}


def load_song(args):
    root = pathlib.Path(args.song)
    if not root.exists():
        root = HERE / "songs" / args.song
    song = song_module.load(root)
    step = song.step(args.step) if args.step else song.steps[0]
    if step.carry_from is not None:
        raise SystemExit("the microscope observes one standalone step; this step carries from another")
    if args.seed is not None:
        step.seed = args.seed
    if args.score is not None:
        step.score_file = str(pathlib.Path(args.score).resolve())
    if args.seconds is not None:
        step.seconds, step.max_tokens = args.seconds, None
    if args.profile is not None:
        song.pipeline = {**song.pipeline, "profile": args.profile}
    song.label = args.label or f"{song.id}.{step.id}.s{step.seed}"
    return song, step


def stage_hashes(take) -> dict:
    return {**take.stages, "pcm_frames": len(take.audio)}


def cmd_capture(args):
    run = pathlib.Path(args.run).expanduser().resolve()
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"{run} is not empty; every run gets its own directory")
    song, step = load_song(args)
    run.mkdir(parents=True, exist_ok=True)

    request = renderer.request_for(song, step)
    (run / "prompt").mkdir()
    (run / "prompt" / "style.txt").write_text(request.style)
    (run / "prompt" / "lyrics.txt").write_text(request.lyrics)
    scope.write_json(run / "prompt" / "request.json", request.to_dict())

    observe = not args.off
    recorder = scope.Recorder(ode_checkpoints=parse_steps(args.ode_checkpoints), capture=args.capture) \
        if observe else None
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    pipe = renderer.build_pipeline(pathlib.Path(args.models), song, progress=not args.quiet)
    take = renderer.render_step(pipe, song, step,
                                on_token=recorder.on_token if observe else None,
                                on_step=recorder.on_step if observe else None)
    take.write(run / "take")
    scope.write_json(run / "baseline" / "hashes.json", stage_hashes(take))
    print("\n  " + "  ".join(f"{k} {v['hash']}" for k, v in take.stages.items()))

    metadata = {
        "run": run.name, "started": started, "finished": None, "microscope": observe,
        "invocation": {"argv": sys.argv, "executable": sys.executable, "cwd": str(pathlib.Path.cwd())},
        "repos": {"audiogen-yue2": git_commit(HERE), "engine": engine_source(),
                  "audiogen-comfyui": git_commit(HERE.parent / "audiogen-comfyui")},
        "machine": machine(),
        "microscope_config": {"capture": args.capture, "ode_checkpoints": args.ode_checkpoints,
                              "decode_steps": not args.no_decode, "listening_copies": not args.no_listening}
        if observe else None,
        "engine_config": json.loads((run / "take" / "config.json").read_text()),
        "provenance": renderer.provenance(song, [take], pathlib.Path(args.models)),
    }
    scope.write_json(run / "metadata.json", metadata)

    if observe:
        report = scope.write_capture(run, pipe, take, recorder, rate=RATE)
        final_dir = run / "final"
        final_dir.mkdir(exist_ok=True)
        np.save(final_dir / "latent.npy", np.asarray(take.latents, dtype=np.float32))
        scope.write_wav(final_dir / "audio.wav", take.audio, RATE)
        # The last captured state of a single-chunk solve must BE the returned latent.
        if len(recorder.flow) == 1 and 0 in recorder.flow:
            states = recorder.states(0)
            last = states[max(states)]
            report["integrity"]["final_state_equals_latent"] = bool(
                max(states) == recorder.flow[0]["meta"]["steps"] and np.array_equal(last, take.latents))
        if not args.no_decode:
            rows = scope.decode_flow(run, pipe, recorder, take.audio, rate=RATE,
                                     listening=not args.no_listening)
            if 0 in rows and rows[0] and rows[0][-1]["step"] == recorder.flow[0]["meta"]["steps"]:
                report["integrity"]["final_state_decodes_to_pcm"] = \
                    rows[0][-1]["pcm_hash"] == take.stages["pcm"]["hash"]
        alignment = score_module.alignment(take.score or "", request.lyrics)
        scope.write_json(run / "analysis" / "alignment.json", alignment)
        (run / "analysis" / "alignment.txt").write_text(score_module.alignment_report(alignment))
        scope.write_json(run / "analysis" / "annotations.json", scope.annotation_template(recorder))
        scope.write_json(run / "analysis" / "capture.json", report)
        (run / "analysis" / "metrics.csv").write_text(scope.metrics_csv(run))
        write_timeline(run)
        print("  integrity: " + ", ".join(f"{k}={v}" for k, v in report["integrity"].items()))

    metadata["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    scope.write_json(run / "metadata.json", metadata)
    print(f"wrote {run}")
    return 0 if not observe or all(v is not False for v in report["integrity"].values()) else 1


def parse_steps(spec):
    return "all" if spec in (None, "all") else [int(s) for s in str(spec).split(",") if s.strip()]


def decoded_pcm(path: pathlib.Path) -> np.ndarray:
    import soundfile
    audio, _ = soundfile.read(path, dtype="float32", always_2d=True)
    return audio


def run_hashes(run: pathlib.Path) -> dict:
    """Stage hashes from a microscope run, or from a bin/render.py output directory."""
    if (run / "baseline" / "hashes.json").exists():
        return json.loads((run / "baseline" / "hashes.json").read_text())
    provenance = next(run.rglob("provenance.json"), None)
    if provenance is None:
        raise SystemExit(f"{run}: neither baseline/hashes.json nor a provenance.json")
    return json.loads(provenance.read_text())["takes"][0]["stages"]


def run_flac(run: pathlib.Path) -> pathlib.Path:
    return next(p for p in sorted(run.rglob("audio.flac")))


def cmd_compare(args):
    a, b = (pathlib.Path(p).expanduser() for p in (args.a, args.b))
    ha, hb = run_hashes(a), run_hashes(b)
    same = True
    for stage in ("abc", "semantic", "latent", "pcm"):
        x, y = ha.get(stage, {}).get("hash"), hb.get(stage, {}).get("hash")
        same &= x == y
        print(f"  {stage:<9} {x}  {y}  {'same' if x == y else 'DIFFERENT'}")
    # The FLAC files embed nothing that varies here, but the rule is to compare
    # decoded samples, never container bytes.
    pa, pb = decoded_pcm(run_flac(a)), decoded_pcm(run_flac(b))
    pcm = pa.shape == pb.shape and np.array_equal(pa, pb)
    print(f"  decoded FLAC samples {pa.shape} vs {pb.shape}: {'identical' if pcm else 'DIFFERENT'}")
    if not pcm and pa.shape == pb.shape:
        diff = np.abs(pa.astype(np.float64) - pb)
        first = int(np.argmax(diff.max(axis=1) > 0))
        print(f"    first differing frame {first} ({first / RATE:.3f}s), max |diff| {diff.max():.3g}")
    return 0 if same and pcm else 1


def load_take(run: pathlib.Path):
    from yue2.pipeline import SymbolicPlan
    take = run / "take"
    plan = SymbolicPlan.load(take)
    semantic = np.load(take / "semantic.npy").astype(int).tolist()
    return plan, semantic


def song_for_run(run: pathlib.Path, models: str):
    """Rebuild the pipeline a run used, from its own recorded manifest."""
    meta = json.loads((run / "metadata.json").read_text())
    argv = meta["invocation"]["argv"]
    song_arg = argv[argv.index("capture") + 1]
    root = pathlib.Path(song_arg) if pathlib.Path(song_arg).exists() else HERE / "songs" / song_arg
    song = song_module.load(root)
    song.pipeline = {**song.pipeline, **(meta["provenance"].get("pipeline") or {})}
    return song, meta


def synthesize_decode(pipe, semantic, **kwargs):
    latents = pipe.synthesize(semantic, **kwargs)
    return latents, pipe.decode(latents)


def cmd_prefixes(args):
    from yue2.pipeline import SemanticResult
    run = pathlib.Path(args.run).expanduser().resolve()
    plan, tokens = load_take(run)
    song, _ = song_for_run(run, args.models)
    pipe = renderer.build_pipeline(pathlib.Path(args.models), song, progress=not args.quiet)
    final = decoded_pcm(run / "final" / "audio.wav") if (run / "final" / "audio.wav").exists() else None
    marks = [int(t) for t in args.tokens.split(",")]
    rows = []
    for mark in marks:
        if not 0 < mark <= len(tokens):
            print(f"  skip {mark}: the take has {len(tokens)} tokens")
            continue
        # A real prefix, nothing appended and nothing padded: the solve sees
        # exactly these tokens and the same seed, hence the same noise frames.
        semantic = SemanticResult(plan, tokens[:mark], {}, False)
        latents, audio = synthesize_decode(pipe, semantic)
        name = f"{mark:06d}"
        (run / "semantic" / "prefix_audio").mkdir(parents=True, exist_ok=True)
        np.save(run / "semantic" / "prefix_audio" / f"{name}.latent.npy", latents)
        scope.write_wav(run / "semantic" / "prefix_audio" / f"{name}.wav", audio, RATE)
        row = {"tokens": mark, "seconds_of_music": mark / scope.SEMANTIC_RATE, "frames": len(audio),
               "latent_hash": hashes.hash_array(latents), "pcm_hash": hashes.hash_array(audio),
               "padding": None}
        if final is not None:
            head = final[:len(audio)]
            row["same_span_of_final"] = scope.audio_metrics(audio, head, RATE)
            full_latent = np.load(run / "final" / "latent.npy")[:mark]
            row["latent_equals_final_head"] = bool(np.array_equal(latents, full_latent))
            row["latent_delta_final_head_relative"] = float(
                np.linalg.norm(latents.astype(np.float64) - full_latent) / (np.linalg.norm(full_latent) or 1))
        rows.append(row)
        print(f"  {mark:>6} tokens  {mark / 25:7.2f}s  corr-with-final-head "
              f"{row.get('same_span_of_final', {}).get('correlation_final')}")
    scope.write_json(run / "semantic" / "prefix_audio" / "analysis.json",
                     {"seed": plan.request.seed, "rows": rows,
                      "note": "each prefix solved alone: same plan, same seed, tokens[:N], no padding"})
    return 0


def cmd_ode_steps(args):
    """Final outputs of solves configured for N steps, with the semantic tokens held fixed."""
    from yue2.pipeline import SemanticResult
    run = pathlib.Path(args.run).expanduser().resolve()
    plan, tokens = load_take(run)
    song, _ = song_for_run(run, args.models)
    pipe = renderer.build_pipeline(pathlib.Path(args.models), song, progress=not args.quiet)
    final = decoded_pcm(run / "final" / "audio.wav")
    final_latent = np.load(run / "final" / "latent.npy")
    out = run / "ode_steps"
    rows = []
    default = pipe.generation_config
    for n in [int(s) for s in args.steps.split(",")]:
        pipe.generation_config = dataclasses.replace(default, ode_steps=n)
        latents, audio = synthesize_decode(pipe, SemanticResult(plan, tokens, {}, False))
        scope.write_wav(out / f"steps_{n:03d}.wav", audio, RATE)
        np.save(out / f"steps_{n:03d}.latent.npy", latents)
        row = {"ode_steps": n, "latent_hash": hashes.hash_array(latents), "pcm_hash": hashes.hash_array(audio),
               "latent_equals_captured_32": bool(np.array_equal(latents, final_latent)),
               "latent_delta_relative": float(np.linalg.norm(latents.astype(np.float64) - final_latent)
                                              / (np.linalg.norm(final_latent) or 1)),
               **scope.audio_metrics(audio, final, RATE)}
        rows.append(row)
        print(f"  {n:>3} steps  corr {row.get('correlation_final')}  latent Δ {row['latent_delta_relative']:.4f}")
    pipe.generation_config = default
    scope.write_json(out / "analysis.json", {
        "note": "final outputs of solves configured for N steps; not the same states as step N of a 32-step solve",
        "semantic_hash": hashes.hash_tokens(tokens), "seed": plan.request.seed, "rows": rows})
    return 0


def cmd_annotate(args):
    run = pathlib.Path(args.run).expanduser()
    path = run / "analysis" / "annotations.json"
    notes = json.loads(path.read_text())
    if args.phrase:
        notes.setdefault("phrases", {})[args.phrase] = " ".join(args.values)
    else:
        chunk = notes["ode_predicted" if args.predicted else "ode"][f"chunk_{args.chunk:03d}"][str(args.step)]
        for pair in args.values:
            key, _, value = pair.partition("=")
            if key not in scope.LISTENING:
                raise SystemExit(f"unknown judgment {key!r}; one of {', '.join(scope.LISTENING)}")
            chunk[key] = {"true": True, "false": False, "null": None, "": None}[value.lower()]
    scope.write_json(path, notes)
    write_timeline(run)
    return 0


def write_timeline(run: pathlib.Path):
    events = scope.timeline(run)
    scope.write_json(run / "analysis" / "timeline.json", events)
    (run / "analysis" / "timeline.md").write_text(scope.timeline_markdown(run, events))


def cmd_timeline(args):
    run = pathlib.Path(args.run).expanduser()
    write_timeline(run)
    print((run / "analysis" / "timeline.md").read_text())
    return 0


def cmd_study(args):
    """Run each arm of a manifest as its own capture, then compare them.

    {"kind": "seed" | "fixed_abc" | "fixed_semantic", "song": "burn_it_down",
     "out": "<dir>", "seeds": [...], "score_from": "<run>", "semantic_from": "<run>",
     "ode_steps": [...], "capture": {"ode_checkpoints": "0,8,16,24,32", "no_decode": true}}
    """
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    out = pathlib.Path(manifest["out"]).expanduser()
    kind = manifest["kind"]
    base = [sys.executable, str(pathlib.Path(__file__).resolve())]
    capture = manifest.get("capture", {})
    extra = [f"--ode-checkpoints={capture.get('ode_checkpoints', 'all')}"]
    extra += ["--no-decode"] if capture.get("no_decode") else []
    extra += ["--no-listening"] if capture.get("no_listening") else []
    extra += ["--quiet"] if args.quiet else []
    if kind in ("seed", "fixed_abc"):
        score = None
        if kind == "fixed_abc":
            score = pathlib.Path(manifest["score_from"]).expanduser() / "take" / "score.abc"
            if not score.exists():
                raise SystemExit(f"{score} missing")
        for seed in manifest["seeds"]:
            run = out / f"{manifest['song']}.{kind}.s{seed}"
            if (run / "metadata.json").exists():
                print(f"  have {run.name}")
                continue
            cmd = base + ["capture", manifest["song"], "--run", str(run), "--seed", str(seed)] + extra
            cmd += ["--score", str(score)] if score else []
            print("  $", " ".join(cmd))
            subprocess.run(cmd, check=True)
    elif kind == "fixed_semantic":
        source = pathlib.Path(manifest["semantic_from"]).expanduser()
        steps = ",".join(str(s) for s in manifest.get("ode_steps", [8, 16, 32, 64]))
        subprocess.run(base + ["ode-steps", str(source), "--steps", steps], check=True)
    else:
        raise SystemExit(f"unknown study kind {kind!r}")
    if kind != "fixed_semantic":
        summarise_study(out, manifest)
    return 0


def summarise_study(out: pathlib.Path, manifest: dict):
    rows = []
    for run in sorted(out.glob(f"{manifest['song']}.{manifest['kind']}.s*")):
        h = run_hashes(run)
        align = json.loads((run / "analysis" / "alignment.json").read_text()) \
            if (run / "analysis" / "alignment.json").exists() else {}
        rows.append({"run": run.name, "seed": json.loads((run / "prompt" / "request.json").read_text())["seed"],
                     **{f"{k}_hash": h.get(k, {}).get("hash") for k in ("abc", "semantic", "latent", "pcm")},
                     "semantic_tokens": h.get("semantic", {}).get("tokens"),
                     "notes_per_syllable": align.get("song", {}).get("notes_per_syllable"),
                     "vocal_notes": align.get("song", {}).get("melody_notes"),
                     "bins": align.get("bins"),
                     "manual_intelligibility": json.loads((run / "analysis" / "annotations.json").read_text())
                     .get("phrases") if (run / "analysis" / "annotations.json").exists() else None})
    scope.write_json(out / f"{manifest['kind']}-summary.json", {"manifest": manifest, "runs": rows})
    print(f"  summary: {out / (manifest['kind'] + '-summary.json')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", default=str(HERE / "models"))
    common.add_argument("--quiet", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("capture", parents=[common])
    c.add_argument("song")
    c.add_argument("--run", required=True, help="new, empty run directory")
    c.add_argument("--step")
    c.add_argument("--seed", type=int)
    c.add_argument("--score", help="supply this ABC instead of planning one")
    c.add_argument("--seconds", type=float)
    c.add_argument("--label")
    c.add_argument("--profile", choices=("official", "comfyui-yue2-mps-v1"))
    c.add_argument("--off", action="store_true", help="render without observing, for ON/OFF comparison")
    c.add_argument("--capture", choices=("cpu", "reference"), default="cpu")
    c.add_argument("--ode-checkpoints", default="all", help="'all' or e.g. 0,1,2,4,8,16,32")
    c.add_argument("--no-decode", action="store_true", help="keep ODE latents, skip decoding each")
    c.add_argument("--no-listening", action="store_true", help="skip the peak-normalised copies")
    c.set_defaults(fn=cmd_capture)

    c = sub.add_parser("compare", parents=[common])
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=cmd_compare)

    c = sub.add_parser("prefixes", parents=[common])
    c.add_argument("run")
    c.add_argument("--tokens", default="500,1000,2000,4000")
    c.set_defaults(fn=cmd_prefixes)

    c = sub.add_parser("ode-steps", parents=[common])
    c.add_argument("run")
    c.add_argument("--steps", default="1,2,4,8,12,16,24,32,48,64")
    c.set_defaults(fn=cmd_ode_steps)

    c = sub.add_parser("annotate", parents=[common])
    c.add_argument("run")
    c.add_argument("--chunk", type=int, default=0)
    c.add_argument("--step", type=int)
    c.add_argument("--phrase", help="e.g. verse.1.2 (section label, occurrence, phrase)")
    c.add_argument("--predicted", action="store_true",
                   help="judge the predicted-final audio at --step rather than the ODE state")
    c.add_argument("values", nargs="+", help="key=true|false|null, or free text for --phrase")
    c.set_defaults(fn=cmd_annotate)

    c = sub.add_parser("timeline", parents=[common])
    c.add_argument("run")
    c.set_defaults(fn=cmd_timeline)

    c = sub.add_parser("study", parents=[common])
    c.add_argument("manifest")
    c.set_defaults(fn=cmd_study)

    args = ap.parse_args()
    if args.command == "annotate" and args.step is None and not args.phrase:
        raise SystemExit("annotate needs --step or --phrase")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
