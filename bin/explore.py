#!/usr/bin/env python
"""Interactive 3D explorers over finished microscope runs (local HTML, three.js vendored).

    bin/explore.py focus RUN           "Coming into focus": the ODE as a spectrogram terrain, per step
    bin/explore.py stack RUN           "From plan to sound": score / semantic tokens / latent / audio on one time axis
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
import functools
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


def aac_args(encoder: str, bitrate: str) -> list[str]:
    """ffmpeg's own AAC encoder ("aac") or Apple's AudioToolbox one ("aac_at", macOS only)."""
    return ["-c:a", encoder, "-b:a", bitrate, "-map_metadata", "-1", "-fflags", "+bitexact", "-movflags", "+faststart"]


@functools.cache
def _ffmpeg_version() -> bytes:
    import subprocess
    return subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.split(b"\n")[0]


def _publish_audio(value, out: pathlib.Path, base: str, done: dict, enc: list[str]):
    """Replace every audio path in a data tree with its AAC copy under out/media/."""
    import hashlib
    import subprocess
    if isinstance(value, dict):
        return {k: _publish_audio(v, out, base, done, enc) for k, v in value.items()}
    if isinstance(value, list):
        return [_publish_audio(v, out, base, done, enc) for v in value]
    if not (isinstance(value, str) and value.endswith(AUDIO_SUFFIXES)):
        return value
    if value not in done:
        source = (out / value).resolve()
        # Named by what makes the bytes: the source, the encoder settings and the encoder's version. The URLs are
        # cached forever, so a change to any of them must give a new name.
        digest = hashlib.sha256(source.read_bytes() + " ".join(enc).encode() + _ffmpeg_version()).hexdigest()[:16]
        name = f"{digest}.m4a"
        target = out / "media" / name
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), *enc, str(target)], check=True)
        done[value] = name
    return f"{base}/{done[value]}"


def _version_assets(out: pathlib.Path) -> None:
    """Stamp each page's script and stylesheet URLs with their content hash, so a browser holding an older copy (GitHub
    Pages lets them cache for 10 minutes) fetches the new one as soon as the page does."""
    import hashlib
    import re

    def stamp(m):
        path = out / m.group(2)
        return f'{m.group(1)}{m.group(2)}?v={hashlib.sha256(path.read_bytes()).hexdigest()[:10]}"' if path.exists() else m.group(0)
    for page in out.glob("*.html"):
        page.write_text(re.sub(r'((?:src|href)=")((?:app|data|vendor)/[^"?]+\.js|explore\.css)"', stamp, page.read_text()))


def cmd_publish(args):
    """One take's Coming into focus and From plan to sound, as a site that can go on the web."""
    import json
    import shutil
    run = pathlib.Path(args.run).resolve()
    if not explore.is_finished(run):
        sys.exit(f"{run}: metadata.json has no 'finished'")
    if run != DEFAULT_RUN.resolve() and args.title is None:
        sys.exit("name the song for the page: --title (and --take)")
    if args.title is None:
        args.title, args.take = "Slow Down", args.take or 2
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
    focus["annotate"] = None                          # nothing to write to on the web: a visitor's marks stay in their browser
    focus["listening"] = {**focus["listening"], "state": {}, "predicted": {}}   # visitors start from the machines' marks
    focus["run"] = stack["run"] = label
    # Every file twice: standard (small, the default) and HD, which the pages' "HD audio" switch swaps in.
    done: dict = {}
    done_hd: dict = {}
    enc, enc_hd = aac_args(args.aac_encoder, args.aac_bitrate), aac_args(args.hd_encoder, args.hd_bitrate)
    _publish_audio(focus, out, args.media_base, done_hd, enc_hd)
    _publish_audio(stack, out, args.media_base, done_hd, enc_hd)
    focus = _publish_audio(focus, out, args.media_base, done, enc)
    stack = _publish_audio(stack, out, args.media_base, done, enc)
    size = lambda names: sum((out / "media" / n).stat().st_size for n in names)   # noqa: E731
    focus_src = [k for k in done if f"{args.media_base}/{done[k]}" in
                 {focus["audio"]["finished"], *focus["audio"]["state"], *focus["audio"]["predicted"]}]
    # The page downloads all of it before playing; the totals let its progress bar count bytes.
    focus["audio_bytes"] = size({done[k] for k in focus_src})
    hd = {"map": {f"{args.media_base}/{done[k]}": f"{args.media_base}/{done_hd[k]}" for k in done},
          "bytes": size({done_hd[k] for k in focus_src}), "rate": args.hd_bitrate, "standard_rate": args.aac_bitrate}
    focus["hd"] = stack["hd"] = hd
    # The finished song at HD quality, under a readable name, for the pages' download button. The upload serves it with
    # Content-Disposition: attachment (see media-manifest.json "attachments"), because a cross-origin link can't
    # rename the file itself.
    import re
    fin = next(k for k in done if f"{args.media_base}/{done[k]}" == focus["audio"]["finished"])
    dl_name = f"{re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')}-{done_hd[fin][:8]}.m4a"
    shutil.copyfile(out / "media" / done_hd[fin], out / "media" / dl_name)
    dl_file = f"{args.title} - YuE2{f' take {args.take}' if args.take else ''}.m4a"
    focus["download"] = stack["download"] = {"url": f"{args.media_base}/{dl_name}", "file": dl_file,
                                             "bytes": size({dl_name}), "rate": args.hd_bitrate}
    explore.write_data(out, "focus", "FOCUS", focus)
    explore.write_data(out, "stack", "STACK", stack)
    explore.write_data(out, "summary", "SUMMARY", {
        "focus": {"run": label, "steps": focus["steps"]},
        "stack": {"run": label, "score_seconds": stack["score"]["seconds"], "audio_seconds": stack["seconds"]},
        "built": datetime.date.today().isoformat()})
    # No take map on the web: without every take it says nothing.
    _swap(out / "app" / "common.js", ", ['map.html', 'Take map', 'Map']", "")
    _version_assets(out)
    index = out / "index.html"
    _swap(index, "YuE2 generation microscope", "An exploration of how YuE2 generates songs")
    _swap(index, "<p>Three interactive 3D views: one take of one song resolving from noise, the four layers the model writes for it on one\n"
                 "       time axis, and every finished take of another song placed by how its sound moves. Each card names its song.",
          f"<p>Two interactive 3D views of one song, <b>{args.title}</b>, as YuE2, an open-weight AI music model, makes it: the sound\n"
          "       resolving from noise, and the song in four forms from plan to sound, on one time axis. The pages show the saved\n"
          "       states of one real generation; the plan-to-sound replay is an illustration. Audio previews are peak-normalized and compressed for the web.")
    start = index.read_text().index('    <a class="card" href="map.html">')
    end = index.read_text().index("</a>", start) + len("</a>\n")
    index.write_text(index.read_text()[:start] + index.read_text()[end:])
    _swap(index, ":root { --cols: 3; }", ":root { --cols: 2; }")
    _swap(index, '<div class="panel"><h3>Build</h3><ul id="build"></ul></div>',
          f'<div class="panel"><h3>About</h3><ul><li>An exploration of how YuE2 generates songs. {args.about}</li>'
          f'<li><a href="https://github.com/multimodal-art-projection/YuE">YuE2</a> is made by the <a href="https://huggingface.co/m-a-p">Multimodal Art Projection (m-a-p)</a> '
          f'team (<a href="https://huggingface.co/m-a-p/YuE2-3B">model</a>, <a href="https://github.com/multimodal-art-projection/YuE/blob/main/docs/technical_report.pdf">technical '
          f'report</a>). This is not their project: it was run and recorded on a Mac with a '
          f'<a href="https://github.com/jeremy-boschen/YuE">fork of their inference library</a>.</li>'
          f'<li>Download the song: <a href="{focus["download"]["url"]}" download>{focus["download"]["file"]}</a> (AAC {args.hd_bitrate})</li>'
          f'<li>Made by <a href="https://github.com/jeremy-boschen">Jeremy Boschen</a>, built with Claude. '
          f'Source code: <a href="{args.source}">{args.source.split("//")[-1]}</a></li>'
          f'<li><a href="{args.home}">{args.home.split("//")[-1].rstrip("/")}</a></li></ul></div>')
    text = index.read_text()
    start = text.index("  document.getElementById('build').innerHTML")
    end = text.index("`;\n", start) + len("`;\n")
    index.write_text(text[:start] + text[end:])
    manifest = sorted({*done.values(), *done_hd.values(), dl_name})
    (out / "media-manifest.json").write_text(json.dumps(
        {"base": args.media_base, "files": manifest, "attachments": {dl_name: dl_file},
         "bytes": sum((out / "media" / n).stat().st_size for n in manifest)}, indent=1) + "\n")
    found = [(p.relative_to(out), leak) for p in out.rglob("*") if p.suffix in (".html", ".js", ".css", ".json")
             and p.parent.name != "three" for leak in LEAKS if leak in p.read_text(errors="ignore")]
    if found:
        sys.exit(f"local details would be published: {found}")
    print(f"wrote {out}: {len(manifest)} audio files, "
          f"{sum((out / 'media' / n).stat().st_size for n in manifest) / 2**20:.0f} MB, media base {args.media_base}")


def annotate(out: pathlib.Path, body: dict) -> dict:
    """Write one listening answer into a run's analysis/annotations.json (the focus page's M)."""
    import json
    import os
    from audiogen import microscope as scope
    run = (out / str(body.get("run", ""))).resolve()          # the page's own relative path
    path = run / "analysis" / "annotations.json"
    if not run.is_relative_to(out.parent.resolve()) or not path.is_file():
        raise ValueError("no such run")
    view = {"state": "ode", "predicted": "ode_predicted"}.get(body.get("view"))
    field, value, step = body.get("field"), body.get("value"), str(body.get("step"))
    if view is None or field not in scope.LISTENING or value not in (True, False, None):
        raise ValueError("view must be state or predicted, field a listening question, value true, false or null")
    notes = json.loads(path.read_text())
    answers = notes[view]["chunk_000"].get(step)
    if answers is None:
        raise ValueError(f"no step {step}")
    answers[field] = value
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(notes, indent=2) + "\n")
    os.replace(partial, path)
    return explore.listening_marks(run)


def cmd_serve(args):
    """Serve the microscope-runs directory on localhost and open the explorer in it."""
    import functools
    import http.server
    import json
    import webbrowser
    root = args.out.parent          # the pages reach the runs with ../

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/api/annotate":
                return self.send_error(404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                reply, code = json.dumps(annotate(args.out, body)).encode(), 200
            except (ValueError, KeyError, TypeError) as exc:
                reply, code = json.dumps({"error": str(exc)}).encode(), 400
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    handler = functools.partial(Handler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/{args.out.name}/index.html"
    print("serving", root, "at", url, flush=True)
    if not args.no_open:
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
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8771)
    serve.add_argument("--no-open", action="store_true", help="serve without opening a browser (the page tests use this)")
    publish = sub.add_parser("publish", help="one take's focus and stack views as a site for the web")
    publish.add_argument("run", nargs="?", default=str(DEFAULT_RUN))
    publish.add_argument("--title", help="the song's name; required unless publishing the default run")
    publish.add_argument("--take", type=int)
    publish.add_argument("--aac-encoder", default="aac_at", choices=["aac", "aac_at"], help="standard audio (the default)")
    publish.add_argument("--aac-bitrate", default="64k")
    publish.add_argument("--hd-encoder", default="aac_at", choices=["aac", "aac_at"], help="HD audio (the switch)")
    publish.add_argument("--hd-bitrate", default="160k")
    publish.add_argument("--media-base", default="media",
                         help="where the audio will be served from: 'media' (beside the pages) or a URL")
    publish.add_argument("--home", default="https://www.newty.coffee/")
    publish.add_argument("--source", default="https://github.com/jeremy-boschen/audiogen-yue2")
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
