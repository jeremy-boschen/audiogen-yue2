"""Drive the pipeline stage by stage, so takes can carry each other.

The library's __call__ runs plan -> semantic -> synthesize -> decode in one go,
which is the right shape for a single take and the wrong one for a chain: a
continuation has to hand the previous take's semantic tokens to generation and
its latents to synthesis. So the stages are walked here instead, and every
boundary is hashed on the way past.
"""
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hashes, lora
from .song import PIPELINE_KEYS, Song, Step, seconds_to_tokens

SAMPLE_RATE = 48000


@dataclass
class Take:
    """One rendered step, and everything the next one might need from it."""
    step_id: str
    semantic: list
    latents: np.ndarray
    audio: np.ndarray
    score: str
    stages: dict
    seconds: float
    render_seconds: float
    plan_prefix_tokens: int = 0
    carried: dict = field(default_factory=dict)
    result: object = None                  # the engine's SongResult, for its own writer

    def write(self, directory: Path) -> Path:
        """Write the take in the engine's native artifact layout.

        The layout matters beyond tidiness: the skill's listening page
        (YuE/skills/yue2-music/scripts/listen.py) re-hashes every file it copies
        against the `artifacts` block in result.json and withholds the player
        when they disagree. That check is only worth anything if the receipt was
        written by whatever produced the audio, so it is written here, from the
        stage outputs still in memory, rather than reconstructed later from the
        files on disk -- which would only prove the files equal themselves.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.result.save_artifacts(directory)
        return directory


def build_pipeline(models: Path, song: Song, *, progress: bool = True):
    from yue2.pipeline import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    unknown = set(song.pipeline) - PIPELINE_KEYS
    if unknown:
        raise ValueError(f"unknown pipeline keys {sorted(unknown)}; known: {sorted(PIPELINE_KEYS)}")
    import torch
    options = dict(song.pipeline)
    # An explicit profile wins over the device-specific default.
    if "profile" not in options:
        device = options.get("device", "auto")
        is_mps = device == "mps" or (device == "auto" and not torch.cuda.is_available()
                                      and torch.backends.mps.is_available())
        options["profile"] = "comfyui-yue2-mps-v1" if is_mps else "official"
    if "backend" not in options and not flash_attention_built(torch):
        # The CUDA-graph decode path calls aten::_flash_attention_forward
        # directly, and PyTorch's own Windows wheels are built without
        # FlashAttention -- so it does not fail to be fast, it raises
        # "USE_FLASH_ATTENTION was not enabled for build" partway into the
        # first step. torch-eager takes the other path, which goes through
        # scaled_dot_product_attention and falls back on its own.
        #
        # Probed, not assumed from the platform: a torch built WITH flash on
        # Windows should keep the graphs. And is_flash_attention_available is
        # the right probe -- flash_sdp_enabled() returns True here, because it
        # reports which SDP backend is preferred, not what was compiled in.
        options["backend"] = "torch-eager"
    config = GenerationConfig(**fit_generation_config(song, options["profile"]))
    pipe = YuE2Pipeline(Path(models) / "YuE2-3B", Path(models) / "YuE2-Vae",
                        generation_config=config, progress=progress, **options)
    adapters = lora.from_manifest(song.lora)
    if adapters:
        # Attached through the engine's hook rather than by patching pipe._model:
        # the engine rebuilds and moves that model between stages, so a patch
        # applied once from outside is silently dropped.
        pipe.on_model_ready.append(lora.hook(adapters))
    return pipe


def flash_attention_built(torch) -> bool:
    """Whether this torch can run aten::_flash_attention_forward at all.

    True off CUDA: the question is only asked to decide whether the CUDA graph
    path is safe, and nothing else reaches it.
    """
    if not torch.cuda.is_available():
        return True
    probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    return True if probe is None else bool(probe())


def fit_generation_config(song: Song, profile_name: str) -> dict:
    """Refuse what the profile cannot do; quietly fix what does not matter.

    Two different things get confused here, so they are separated.

    A carry chain is a real capability. Only 'comfyui-yue2-mps-v1' supports it
    and that profile hard-requires Apple Silicon, so on a CUDA box the chain is
    impossible, permanently, and the honest answer is no. The engine does refuse
    it too -- but inside validate_continuation, which is reached on the SECOND
    step, so the song renders its first step, spends the GPU minutes, and only
    then says no. And it says the profile does not support continuation, which
    is true and unhelpful: the reader did not pick the profile, the machine did.

    `rng_device` is not a capability. It decides which device draws the sampling
    noise, and 'official' insists on 'auto'. Every song here was written on the
    Mac with 'cpu', and refusing them all would be pedantry: the seeds do not
    reproduce across the two machines anyway -- different profile, different
    kernels, different hardware. So it is coerced and said out loud, rather than
    thrown back at someone who cannot do anything about it.
    """
    from yue2.profiles import resolve_profile

    profile = resolve_profile(profile_name)
    if profile.supports_continuation:
        return dict(song.generation_config)

    carried = [step.id for step in song.steps if step.carry_from is not None]
    if carried:
        raise ValueError(
            f"this song grows step {carried[0]!r} out of an earlier one, and the "
            f"{profile.name!r} profile cannot continue a take. Continuation needs "
            f"the Metal profile, which only runs on Apple Silicon -- on this "
            f"machine each step has to stand alone. Remove carry_from from "
            f"{', '.join(repr(c) for c in carried)}, or render this song on the Mac.")

    config = dict(song.generation_config)
    rng = config.get("rng_device")
    if rng not in (None, "auto"):
        print(f"note: rng_device={rng!r} is not available under the {profile.name!r} "
              f"profile; using 'auto'. This take will not match the Mac, which it "
              f"was not going to anyway.")
        config["rng_device"] = "auto"
    return config


def native_result(pipe, song: Song, step: Step, plan, semantic, latents, audio, timing):
    """Wrap our stage outputs in the engine's own result object.

    `pipe.generate()` builds this on the way past; driving the stages by hand to
    get a carry chain skips it, so it is rebuilt here from the same pieces. The
    adapters go into the config because the engine's `weights` block pins the
    base checkpoint, which is not what a LoRA take actually ran: without this,
    two takes that differ only by an adapter carry the same request identity.
    """
    from yue2.pipeline import SongResult
    from yue2.storage import identity

    request = plan.request
    config = pipe.effective_config(request, song.abc_sampling or None,
                                   {**song.semantic_sampling, "max_tokens": step.new_tokens})
    adapters = [a.identity() for a in lora.from_manifest(song.lora)]
    config["audiogen"] = {"step": step.id, "adapters": adapters or None,
                          "carried": carried_note(step),
                          "audio_writer": "yue2.pipeline.SongResult", "pcm_bits": 24}
    return SongResult(audio=np.asarray(audio), sample_rate=SAMPLE_RATE, semantic=semantic,
                      latents=np.asarray(latents), config=config, weights=pipe.weights,
                      timing=timing,
                      request_identity=identity({"request": request.to_dict(), "config": config,
                                                 "weights": pipe.weights}))


def carried_note(step: Step) -> dict | None:
    return None if step.carry_from is None else {"from": step.carry_from, "tokens": step.carry_tokens}


def request_id(song: Song, step: Step) -> str:
    """Name the take after what shaped it, adapters included.

    The engine only checks that this is filename-safe -- it never reaches the
    prompt or the token stream, so an arm tag cannot confound a comparison. It
    does reach the listening page, which titles each case by it (listen.py:201
    reads request.json's id and falls back to the directory name), and two arms
    of an A/B that print the same title are not a comparison anyone can read.

    A supplied label replaces the derived name and is made filename-safe.
    """
    if song.label:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", song.label).lstrip("._-")[:180]
        if not safe:
            raise SystemExit(f"--label {song.label!r} has no filename-safe characters")
        return safe
    parts = [f"{a.path.stem}-x{a.strength:g}" for a in lora.from_manifest(song.lora)]
    name = ".".join([f"{song.id}.{step.id}", *parts])
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:180]


def canonical_score(text: str | None) -> str | None:
    """A supplied score in the form the planner writes: trimmed, ending in one newline.

    The planner's own scores always end in exactly one newline, so a score that
    was planned, saved and supplied back re-tokenises to the planner's exact
    ABC ids however it was edited or saved in between. Stripping the newline
    changes the last ABC token, and with it the whole take.
    """
    text = (text or "").strip()
    return text + "\n" if text else None


def request_for(song: Song, step: Step):
    from yue2.protocol import SongRequest
    def text_input(key, fallback):
        filename = getattr(step, key, None)
        return (song.root / filename).read_text() if filename is not None else fallback

    score = text_input("score_file", song.abc)
    if getattr(song, "normalize_score", True):
        abc = canonical_score(score)
    else:
        abc = score.strip() or None if score is not None else None
    # Explicit style and lyric files retain their whitespace.
    style = text_input("style_file", song.style)
    lyrics = text_input("lyrics_file", song.lyrics)
    return SongRequest(style=style, lyrics=lyrics, id=request_id(song, step),
                       seed=step.seed, cot=song.cot, abc=abc, cfg_scale=song.cfg_scale)


class Preview:
    """Asks for "listen to the song so far" while the semantic stage is writing.

    ``wanted()`` is polled every ``every`` tokens once ``min_tokens`` have been
    written (fewer gives the sound stage too little to go on); when it returns True the tokens
    written so far are voiced and decoded, ``deliver(audio, seconds)`` receives the
    result, and the stage carries on from where it paused. The take is unchanged:
    sampling draws from its own generator and KV cache, and the model is handed back
    to the AR stage exactly as the NAR stage found it (tests/test_preview.py).
    """

    def __init__(self, wanted, deliver, every: int = 25, min_tokens: int = 0):
        self.wanted, self.deliver, self.every, self.min_tokens = wanted, deliver, every, min_tokens


def preview_so_far(pipe, plan, tokens: list[int], step: Step, known_latents=None) -> np.ndarray:
    """Voice and decode ``tokens`` (the song so far), then hand the model back to the AR stage."""
    from yue2.pipeline import SemanticResult
    known = None if known_latents is None else np.asarray(known_latents)[:len(tokens)]
    latents = pipe.synthesize(SemanticResult(plan, list(tokens), {}, False), known_latents=known,
                              blend_seconds=step.blend_seconds if known is not None else 0.0,
                              chunk_seconds=step.chunk_seconds, overlap_seconds=step.overlap_seconds)
    audio = np.asarray(pipe.decode(latents), np.float32)
    pipe._stage_boundary()
    pipe._load_model(for_nar=False)          # re-fires on_model_ready for AR: LoRA back to its AR state
    return audio


def render_step(pipe, song: Song, step: Step, previous: Take | None = None, *,
                on_token=None, on_step=None, plan=None, preview: Preview | None = None) -> Take:
    """Render one step, carrying from `previous` when the step asks for it.

    ``on_token`` and ``on_step`` are the engine's own observers, passed through
    untouched: token callbacks for both sampled stages, and every acoustic ODE
    state (see audiogen.microscope). Left as None, nothing is observed.

    ``plan`` skips planning and uses that exact SymbolicPlan: its ABC token IDs
    as generated, which a score supplied as text does not reproduce.
    """
    clock = time.perf_counter()
    stages: dict = {}
    carried: dict = {}

    if plan is None:
        plan = pipe.plan(request=request_for(song, step), abc_sampling=song.abc_sampling or None,
                         on_token=on_token)
    stages["abc"] = {"hash": hashes.hash_tokens(plan.abc_ids), "tokens": len(plan.abc_ids),
                     "prefix_tokens": len(plan.prefix)}

    carry_tokens = None
    known_latents = None
    if step.carry_from is not None:
        if previous is None or previous.step_id != step.carry_from:
            raise ValueError(f"step {step.id!r} needs {step.carry_from!r} rendered first")
        keep = step.carry_tokens
        if keep > len(previous.semantic):
            raise ValueError(f"step {step.id!r} keeps {keep} tokens of {step.carry_from!r}, "
                             f"which produced only {len(previous.semantic)}")
        carry_tokens = list(previous.semantic[:keep])
        known_latents = np.asarray(previous.latents)[:keep]
        carried = {"from": step.carry_from, "seconds": round(keep / 25, 2), "tokens": keep,
                   "semantic_hash": hashes.hash_tokens(carry_tokens),
                   "latent_hash": hashes.hash_array(known_latents)}

    sampling = {**song.semantic_sampling, "max_tokens": step.new_tokens}
    semantic_observer = on_token
    if preview is not None:
        if pipe.quantization != "none":
            raise ValueError("listen-so-far needs quantization 'none': the fp8 AR preparation "
                             "is not shown to survive a pause for the NAR stage")
        from yue2.protocol import CODEC_OFFSET, CODEC_SIZE
        written: list[int] = list(carry_tokens or [])

        def semantic_observer(phase, token):
            if on_token is not None:
                on_token(phase, token)
            value = int(token) - CODEC_OFFSET
            if 0 <= value < CODEC_SIZE:
                written.append(value)
                if len(written) >= preview.min_tokens and len(written) % preview.every == 0 and preview.wanted():
                    preview.deliver(preview_so_far(pipe, plan, written, step, known_latents), len(written) / 25)
    semantic = pipe.generate_semantic(plan, sampling=sampling, carry=carry_tokens, on_token=semantic_observer)
    stages["semantic"] = {"hash": hashes.hash_tokens(semantic.tokens),
                          "tokens": len(semantic.tokens),
                          "seconds": round(len(semantic.tokens) / 25, 2)}

    if carry_tokens:
        head = list(semantic.tokens[:len(carry_tokens)])
        if head != carry_tokens:
            raise AssertionError(f"step {step.id!r}: carried tokens were not reproduced verbatim; "
                                 "the continuation did not take")

    latents = pipe.synthesize(semantic, known_latents=known_latents,
                              blend_seconds=step.blend_seconds,
                              chunk_seconds=step.chunk_seconds,
                              overlap_seconds=step.overlap_seconds,
                              **({"on_step": on_step} if on_step is not None else {}))
    stages["latent"] = {"hash": hashes.hash_array(latents), "shape": list(np.shape(latents))}

    audio = pipe.decode(latents)
    stages["pcm"] = {"hash": hashes.hash_array(audio), "shape": list(np.shape(audio))}

    render_seconds = round(time.perf_counter() - clock, 1)
    timing = {"abc": plan.timing, "semantic": semantic.timing,
              "load": dict(pipe.load_timing), "e2e_seconds": render_seconds}
    result = native_result(pipe, song, step, plan, semantic, latents, audio, timing)

    return Take(step_id=step.id, semantic=list(semantic.tokens), latents=np.asarray(latents),
                audio=np.asarray(audio), score=plan.abc, stages=stages,
                seconds=round(len(semantic.tokens) / 25, 2),
                render_seconds=render_seconds,
                plan_prefix_tokens=len(plan.prefix), carried=carried, result=result)


# Variables that can change what gets rendered: threading, backend selection,
# device visibility, and which weights HF_HOME/HF_HUB_OFFLINE resolve to.
ENV_PREFIXES = ("PYTORCH", "TORCH", "MPS", "MTLFLASH", "OMP", "MKL", "CUDA", "HF_")
SECRETISH = ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL")


def recorded_env() -> dict:
    """The environment that shapes a render, with credentials named but not copied.

    HF_TOKEN sits squarely inside the HF_ prefix and is a live credential. A
    provenance file is copied into every output directory and travels with the
    audio, so the value must never land in one; that it was set is the part that
    matters for reproducing a run.
    """
    out = {}
    for key in sorted(os.environ):
        if not key.startswith(ENV_PREFIXES):
            continue
        out[key] = "<set, not recorded>" if any(w in key.upper() for w in SECRETISH) else os.environ[key]
    return out


def wrapper_commit() -> str:
    """Which revision of this package rendered the take.

    provenance already pins the engine and torch. It said nothing about the code
    doing the pinning, which is the part most likely to have moved.
    """
    import subprocess
    try:
        here = Path(__file__).resolve().parent
        show = subprocess.run(["git", "-C", str(here), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, timeout=5)
        return show.stdout.strip() if show.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def provenance(song: Song, takes: list[Take], models: Path) -> dict:
    """What a run needs to be reproducible later, resolved rather than referenced."""
    import torch
    from importlib.metadata import version

    return {
        "song": song.id,
        "rendered": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "audiogen": wrapper_commit(),
            "yue2_infer": version("yue2-infer"),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "device": "cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu",
            "models": str(models),
        },
        # Record invocation details alongside the effective render configuration.
        "invocation": {
            "argv": list(sys.argv),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
            "env": recorded_env(),
        },
        "pipeline": song.pipeline or None,
        "lora": [adapter.identity() for adapter in lora.from_manifest(song.lora)] or None,
        "request": {
            "seed": song.seed, "cot": song.cot, "cfg_scale": song.cfg_scale,
            "generation_config": song.generation_config,
            "abc_sampling": song.abc_sampling, "semantic_sampling": song.semantic_sampling,
            "style_sha": hashes.digest(song.style.encode()),
            "lyrics_sha": hashes.digest(song.lyrics.encode()),
        },
        "takes": [
            {"id": take.step_id, "seed": song.step(take.step_id).seed,
             "seconds": take.seconds, "render_seconds": take.render_seconds,
             "carried": take.carried or None, "stages": take.stages}
            for take in takes
        ],
    }
