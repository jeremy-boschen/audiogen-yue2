#!/usr/bin/env python
"""Interactive 3D explorers over finished microscope runs (local HTML, three.js vendored).

    bin/explore.py focus RUN           "Coming into focus": the ODE as a spectrogram terrain, per step
    bin/explore.py stack RUN           "The stack": score / semantic tokens / latent / audio on one time axis
    bin/explore.py map ROOT [ROOT...]  "Take map": every finished take placed by envelope similarity
    bin/explore.py all                 rebuild all three from the default runs, plus index.html
    bin/explore.py serve               serve the pages on 127.0.0.1 and open them (the focus view needs it)
    bin/explore.py publish [RUN]       one take's focus and stack views as a site for the web

Output goes to --out (default ../audiogen/output/microscope-runs/00_explore/).
Pages load their data from data/*.js and refer to audio in the runs by relative
path. The focus view fetches and decodes its audio (Web Audio), which browsers
refuse from file://, so open the pages through `serve`. `map` is safe to re-run as study runs finish: it
only reads runs whose metadata.json has "finished", and caches features in
OUT/cache/ keyed by file size and mtime.
"""
import argparse
import datetime
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from audiogen import explore  # noqa: E402

RUNS = HERE.parent / "audiogen/output/microscope-runs"
DEFAULT_OUT = RUNS / "00_explore"
DEFAULT_RUN = RUNS / "20260926-slow_down-s4417-take2/on-cpu"   # studio take #233, bit-identical
DEFAULT_ROOTS = [RUNS / "20260926-burn_it_down-s777-baseline", RUNS / "burn_it_down-studies"]


def prepare(out: pathlib.Path) -> pathlib.Path:
    out.mkdir(parents=True, exist_ok=True)
    explore.install_assets(out)
    return out / "cache"


def summary(out: pathlib.Path, **parts) -> None:
    path = out / "data/summary.js"
    current = {}
    if path.exists():
        text = path.read_text()
        start = text.find("{")
        current = explore.json.loads(text[start:text.rstrip().rfind("}") + 1]) if start >= 0 else {}
    current.update(parts)
    current["built"] = datetime.datetime.now().isoformat(timespec="seconds")
    explore.write_data(out, "summary", "SUMMARY", current)


def cmd_focus(args):
    run = pathlib.Path(args.run).resolve()
    if not explore.is_finished(run):
        sys.exit(f"{run}: metadata.json has no 'finished'; not reading a run still being captured")
    cache = prepare(args.out)
    data = explore.focus_data(run, args.out, cache)
    print("wrote", explore.write_data(args.out, "focus", "FOCUS", data))
    summary(args.out, focus={"run": data["run"], "steps": data["steps"]})


def cmd_stack(args):
    run = pathlib.Path(args.run).resolve()
    if not explore.is_finished(run):
        sys.exit(f"{run}: metadata.json has no 'finished'; not reading a run still being captured")
    cache = prepare(args.out)
    data = explore.stack_data(run, args.out, cache)
    print("wrote", explore.write_data(args.out, "stack", "STACK", data))
    summary(args.out, stack={"run": data["run"], "score_seconds": data["score"]["seconds"],
                             "audio_seconds": data["seconds"]})


def cmd_map(args):
    cache = prepare(args.out)
    data = explore.map_data([pathlib.Path(r) for r in args.roots], args.out, cache)
    print("wrote", explore.write_data(args.out, "map", "MAP", data), f"({len(data['takes'])} takes)")
    groups = {}
    for take in data["takes"]:
        groups[take["group"]] = groups.get(take["group"], 0) + 1
    summary(args.out, map={"takes": len(data["takes"]), "groups": groups})


def cmd_all(args):
    for command, extra in ((cmd_focus, {"run": DEFAULT_RUN}), (cmd_stack, {"run": DEFAULT_RUN}),
                           (cmd_map, {"roots": DEFAULT_ROOTS})):
        command(argparse.Namespace(**vars(args), **extra))
    print("open", args.out / "index.html")


AUDIO_SUFFIXES = (".wav", ".flac")
# Anything that names this machine must not reach a published page.
LEAKS = (str(pathlib.Path.home()), "/Users/", "localhost", "127.0.0.1", ".internal", "bin/explore.py")


def _swap(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path.name}: expected exactly one {old[:60]!r}; the page changed, update publish")
    path.write_text(text.replace(old, new))


def _publish_audio(value, out: pathlib.Path, base: str, done: dict):
    """Replace every audio path in a data tree with its AAC copy under out/media/."""
    import hashlib
    import subprocess
    if isinstance(value, dict):
        return {k: _publish_audio(v, out, base, done) for k, v in value.items()}
    if isinstance(value, list):
        return [_publish_audio(v, out, base, done) for v in value]
    if not (isinstance(value, str) and value.endswith(AUDIO_SUFFIXES)):
        return value
    if value not in done:
        source = (out / value).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]   # named by content: cache forever
        name = f"{digest}.m4a"
        target = out / "media" / name
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:a", "aac", "-b:a", "160k",
                            "-map_metadata", "-1", "-fflags", "+bitexact", "-movflags", "+faststart", str(target)],
                           check=True)
        done[value] = name
    return f"{base}/{done[value]}"


def cmd_publish(args):
    """One take's Coming into focus and The stack, as a site that can go on the web."""
    import json
    import shutil
    run = pathlib.Path(args.run).resolve()
    if not explore.is_finished(run):
        sys.exit(f"{run}: metadata.json has no 'finished'")
    out = args.out
    if out.exists():
        if not (out / ".microscope-export").exists():
            sys.exit(f"{out} exists and is not an earlier export; refusing to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / ".microscope-export").write_text("written by bin/explore.py publish; replaced on every export\n")
    explore.install_assets(out)
    (out / "map.html").unlink()
    (out / "app" / "map.js").unlink()
    cache = DEFAULT_OUT / "cache"                       # the same spectrogram cache the local pages use
    cache.mkdir(parents=True, exist_ok=True)
    focus, stack = explore.focus_data(run, out, cache), explore.stack_data(run, out, cache)
    label = f"{args.title}, take {args.take}" if args.take else args.title
    focus["run"] = stack["run"] = label
    done: dict = {}
    focus = _publish_audio(focus, out, args.media_base, done)
    stack = _publish_audio(stack, out, args.media_base, done)
    explore.write_data(out, "focus", "FOCUS", focus)
    explore.write_data(out, "stack", "STACK", stack)
    explore.write_data(out, "summary", "SUMMARY", {
        "focus": {"run": label, "steps": focus["steps"]},
        "stack": {"run": label, "score_seconds": stack["score"]["seconds"], "audio_seconds": stack["seconds"]},
        "built": datetime.date.today().isoformat()})
    # No take map on the web: without every take it says nothing.
    _swap(out / "app" / "common.js", ", ['map.html', 'Take map', 'M']", "")
    index = out / "index.html"
    _swap(index, "YuE2 generation microscope · Burn It Down", f"YuE2 generation microscope · {args.title}")
    _swap(index, "<p>Three interactive 3D views of one take and its siblings: the acoustic solve resolving from noise, the four layers the\n"
                 "       model writes on one time axis, and every finished take placed by how its sound moves.",
          "<p>Two interactive 3D views of one song as an AI music model (YuE2) makes it: the sound resolving from noise, and the\n"
          "       four layers the model writes, on one time axis.")
    start = index.read_text().index('    <a class="card" href="map.html">')
    end = index.read_text().index("</a>", start) + len("</a>\n")
    index.write_text(index.read_text()[:start] + index.read_text()[end:])
    _swap(index, ".grid { display: grid; grid-template-columns: repeat(3, 1fr);", ".grid { display: grid; grid-template-columns: repeat(2, 1fr);")
    _swap(index, '<div class="panel"><h3>Build</h3><ul id="build"></ul></div>',
          f'<div class="panel"><h3>About</h3><ul><li>{args.about}</li>'
          f'<li>Made on a Mac with a local build of YuE2; <a href="{args.home}">{args.home.split("//")[-1].rstrip("/")}</a></li></ul></div>')
    text = index.read_text()
    start = text.index("  document.getElementById('build').innerHTML")
    end = text.index("`;\n", start) + len("`;\n")
    index.write_text(text[:start] + text[end:])
    manifest = sorted({name for name in done.values()})
    (out / "media-manifest.json").write_text(json.dumps(
        {"base": args.media_base, "files": manifest,
         "bytes": sum((out / "media" / n).stat().st_size for n in manifest)}, indent=1) + "\n")
    found = [(p.relative_to(out), leak) for p in out.rglob("*") if p.suffix in (".html", ".js", ".css", ".json")
             and p.parent.name != "three" for leak in LEAKS if leak in p.read_text(errors="ignore")]
    if found:
        sys.exit(f"local details would be published: {found}")
    print(f"wrote {out}: {len(manifest)} audio files, "
          f"{sum((out / 'media' / n).stat().st_size for n in manifest) / 2**20:.0f} MB, media base {args.media_base}")


def cmd_serve(args):
    """Serve the microscope-runs directory on localhost and open the explorer in it."""
    import functools
    import http.server
    import webbrowser
    root = args.out.parent          # the pages reach the runs with ../
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/{args.out.name}/index.html"
    print("serving", root, "at", url, flush=True)
    webbrowser.open(url)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("focus").add_argument("run")
    sub.add_parser("stack").add_argument("run")
    sub.add_parser("map").add_argument("roots", nargs="+")
    sub.add_parser("all")
    sub.add_parser("serve").add_argument("--port", type=int, default=8771)
    publish = sub.add_parser("publish", help="one take's focus and stack views as a site for the web")
    publish.add_argument("run", nargs="?", default=str(DEFAULT_RUN))
    publish.add_argument("--title", default="Slow Down")
    publish.add_argument("--take", type=int, default=2)
    publish.add_argument("--media-base", default="media",
                         help="where the audio will be served from: 'media' (beside the pages) or a URL")
    publish.add_argument("--home", default="https://www.newty.coffee/")
    publish.add_argument("--about", default="One song, written and performed by the model from a score and lyrics. "
                                            "Every view plays the audio it shows.")
    args = parser.parse_args()
    if args.command == "publish" and args.out == DEFAULT_OUT:
        args.out = RUNS / "00_publish"
    args.out = args.out.resolve()
    {"focus": cmd_focus, "stack": cmd_stack, "map": cmd_map, "all": cmd_all,
     "publish": cmd_publish, "serve": cmd_serve}[args.command](args)


if __name__ == "__main__":
    main()
