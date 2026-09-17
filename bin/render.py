#!/usr/bin/env python
"""Render a song from songs/<name>/song.json.

    bin/render.py burn_it_down                 render every step in order
    bin/render.py burn_it_down --step grown    render one step (and whatever it carries from)
    bin/render.py burn_it_down --dry-run       resolve and check the manifest, load nothing
    bin/render.py burn_it_down --out ~/dev/projects/audiogen/takes

Audio lands under --out, never in this repo. The repo holds what defines a song;
the studio holds what it renders to.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from audiogen import render as renderer  # noqa: E402
from audiogen import song as song_module  # noqa: E402


def plan_order(song, target):
    """The steps to render: `target` plus the chain it depends on, in order."""
    if target is None:
        return list(song.steps)
    wanted, cursor = [], song.step(target)
    while True:
        wanted.append(cursor)
        if cursor.carry_from is None:
            break
        cursor = song.step(cursor.carry_from)
    return list(reversed(wanted))


def describe(song, steps):
    print(f"\n{song.id}  seed {song.seed}  cot={song.cot}"
          f"  rng={song.generation_config.get('rng_device', 'auto')}")
    print(f"{'step':12} {'seed':>7} {'target':>9} {'carried':>9} {'new tok':>9}  from")
    for step in steps:
        # Printed in seconds however the step was written, so a manifest using the
        # engine's token units is still legible next to one using seconds.
        carried = step.carry_tokens / 25
        print(f"{step.id:12} {step.seed:>7} {step.target_seconds:>8.1f}s "
              f"{carried:>8.1f}s {step.new_tokens:>9}  {step.carry_from or '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--step", help="render this step and its carry chain")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--models", default=str(HERE / "models"))
    ap.add_argument("--dry-run", action="store_true", help="resolve the manifest, load no weights")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.song)
    if not root.exists():
        root = HERE / "songs" / args.song
    song = song_module.load(root)          # validates before anything expensive happens
    steps = plan_order(song, args.step)
    describe(song, steps)

    if args.dry_run:
        print("\ndry run: manifest resolves and every carry is satisfiable")
        return 0

    out = pathlib.Path(args.out).expanduser() / song.id
    pipe = renderer.build_pipeline(pathlib.Path(args.models), song, progress=not args.quiet)

    takes, previous = [], None
    for step in steps:
        take = renderer.render_step(pipe, song, step, previous)
        take.write(out / step.id)
        takes.append(take)
        previous = take
        marks = "  ".join(f"{name} {value['hash']}" for name, value in take.stages.items())
        print(f"\n{step.id}: {take.seconds}s in {take.render_seconds}s\n  {marks}")

    record = out / "provenance.json"
    record.write_text(json.dumps(renderer.provenance(song, takes, pathlib.Path(args.models)),
                                 indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"provenance {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
