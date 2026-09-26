#!/usr/bin/env python
"""Interactive 3D explorers over finished microscope runs (local HTML, three.js vendored).

    bin/explore.py focus RUN           "Coming into focus": the ODE as a spectrogram terrain, per step
    bin/explore.py stack RUN           "The stack": score / semantic tokens / latent / audio on one time axis
    bin/explore.py map ROOT [ROOT...]  "Take map": every finished take placed by envelope similarity
    bin/explore.py all                 rebuild all three from the default runs, plus index.html

Output goes to --out (default ../audiogen/output/microscope-runs/00_explore/).
Pages load their data from data/*.js (no fetch, so file:// works) and refer to
audio in the runs by relative path; open index.html directly, or serve the
microscope-runs directory. `map` is safe to re-run as study runs finish: it
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
DEFAULT_RUN = RUNS / "20260926-burn_it_down-s777-baseline/on-cpu"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("focus").add_argument("run")
    sub.add_parser("stack").add_argument("run")
    sub.add_parser("map").add_argument("roots", nargs="+")
    sub.add_parser("all")
    args = parser.parse_args()
    args.out = args.out.resolve()
    {"focus": cmd_focus, "stack": cmd_stack, "map": cmd_map, "all": cmd_all}[args.command](args)


if __name__ == "__main__":
    main()
