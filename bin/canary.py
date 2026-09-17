#!/usr/bin/env python
"""Per-stage regression fixture for the yue2 stack.

Renders songs/_canary and hashes every stage boundary, so a change is localised
to the stage that moved instead of showing up as "the audio is different".

    bin/canary.py            verify against songs/_canary/expected.json
    bin/canary.py --record   write that file (only when a change is intended)

Why per stage: the acoustic stage attends over the whole sequence, so a semantic
token that first differs at 200s still changes latent frame 0. An end-to-end
audio hash therefore tells you nothing about where a regression started. It also
makes this the instrument for a cross-machine comparison -- run it on two boxes
and the first stage that disagrees is the answer.
"""
import argparse, hashlib, json, pathlib, sys, time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
SONG = HERE / "songs" / "_canary"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()[:16]


def hash_ids(ids):
    return digest(np.asarray(list(ids), dtype=np.int64).tobytes())


def hash_array(array):
    array = np.ascontiguousarray(array)
    return digest(f"{array.dtype}|{array.shape}|".encode() + array.tobytes())


def build(models, progress):
    from yue2.pipeline import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    spec = json.loads((SONG / "request.json").read_text())
    config = GenerationConfig(**spec.get("generation_config", {}))
    pipe = YuE2Pipeline(models / "YuE2-3B", models / "YuE2-Vae",
                        generation_config=config, progress=progress)
    return pipe, spec


def run(pipe, spec):
    """Walk the stages, hashing each boundary."""
    from yue2.protocol import SongRequest

    style = (SONG / "style.txt").read_text().strip()
    lyrics = (SONG / "lyrics.txt").read_text().strip()
    request = SongRequest(style=style, lyrics=lyrics, id=spec["id"],
                          seed=spec["seed"], cot=spec["cot"], cfg_scale=spec["cfg_scale"])

    stages, clock = {}, time.perf_counter()

    plan = pipe.plan(request=request, abc_sampling=spec["abc_sampling"])
    stages["abc"] = {"hash": hash_ids(plan.abc_ids), "tokens": len(plan.abc_ids),
                     "prefix_tokens": len(plan.prefix)}

    semantic = pipe.generate_semantic(plan, sampling=spec["semantic_sampling"])
    stages["semantic"] = {"hash": hash_ids(semantic.tokens), "tokens": len(semantic.tokens),
                          "seconds": round(len(semantic.tokens) / 25, 2)}

    latents = pipe.synthesize(semantic)
    stages["latent"] = {"hash": hash_array(latents), "shape": list(np.shape(latents))}

    audio = pipe.decode(latents)
    stages["pcm"] = {"hash": hash_array(audio), "shape": list(np.shape(audio))}

    return stages, plan, semantic, latents, audio, round(time.perf_counter() - clock, 1)


def thresholds(stages):
    """What this take does and does not exercise. Recorded so it is not re-derived."""
    tokens = stages["semantic"]["tokens"]
    prefix = stages["abc"]["prefix_tokens"]
    nar_chunk = (24576 - prefix - 3) // 2
    return {
        "min_tokens_200":        {"crossed": tokens > 200,  "note": "EOS suppressed below this (sampling.py)"},
        "penalty_window_50":     {"crossed": tokens > 50,   "note": "repetition window slides past full"},
        "nar_query_block_256":   {"crossed": tokens > 256,  "note": "NAR attention blocks at 256 on MPS/CPU (nar.py)"},
        "nar_multichunk":        {"crossed": tokens > nar_chunk,
                                  "frames_needed": nar_chunk,
                                  "note": "natural NAR split; needs ~%ds at this prefix" % (nar_chunk // 25)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="write expected.json instead of checking it")
    ap.add_argument("--models", default=str(HERE / "models"))
    ap.add_argument("--max-tokens", type=int, help="override for a cheap plumbing check; never record with this")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    pipe, spec = build(pathlib.Path(args.models), progress=not args.quiet)
    if args.max_tokens:
        spec["semantic_sampling"] = {**spec["semantic_sampling"], "max_tokens": args.max_tokens}
        if args.record:
            sys.exit("refusing to record a baseline with --max-tokens; it would not be the fixture")

    stages, plan, semantic, latents, audio, seconds = run(pipe, spec)
    expected_path = SONG / "expected.json"

    if args.record:
        (SONG / "score.abc").write_text(plan.abc)
        np.save(SONG / "reference" / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32))
        np.save(SONG / "reference" / "latent.npy", np.asarray(latents, dtype=np.float32))
        # Audio goes to out/ and is NOT committed: the stage hashes localise a
        # regression better than a waveform, and semantic.npy re-derives it exactly.
        out = HERE / "out"
        out.mkdir(exist_ok=True)
        import soundfile as sf
        sf.write(out / "canary.flac", np.asarray(audio).T if np.ndim(audio) > 1 and
                 np.shape(audio)[0] < np.shape(audio)[-1] else np.asarray(audio),
                 48000, subtype="PCM_24")
        print(f"  audio -> {(out / 'canary.flac').relative_to(HERE)} (gitignored)")
        payload = {"_comment": "Recorded by bin/canary.py --record. Per-stage hashes of the "
                               "fixture render. A change here is a change in engine behaviour.",
                   "recorded": time.strftime("%Y-%m-%d"),
                   "stages": stages, "exercises": thresholds(stages),
                   "render_seconds": seconds}
        expected_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nrecorded {expected_path.relative_to(HERE)} in {seconds}s")
        for name, got in stages.items():
            print(f"  {name:9} {got['hash']}")
        return 0

    if not expected_path.exists():
        sys.exit("no expected.json; run with --record first")
    expected = json.loads(expected_path.read_text())["stages"]

    print(f"\n{'stage':10} {'expected':18} {'got':18} verdict")
    bad = []
    for name in ("abc", "semantic", "latent", "pcm"):
        want, got = expected.get(name, {}).get("hash"), stages[name]["hash"]
        ok = want == got
        if not ok:
            bad.append(name)
        print(f"{name:10} {str(want):18} {got:18} {'ok' if ok else 'DIFFERS'}")

    if not bad:
        print(f"\nidentical through every stage ({seconds}s)")
        return 0
    print(f"\nfirst stage to differ: {bad[0]}")
    print("Everything downstream of it differs as a consequence, not independently.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
