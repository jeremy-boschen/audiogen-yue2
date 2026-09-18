"""Drive the pipeline stage by stage, so takes can carry each other.

The library's __call__ runs plan -> semantic -> synthesize -> decode in one go,
which is the right shape for a single take and the wrong one for a chain: a
continuation has to hand the previous take's semantic tokens to generation and
its latents to synthesis. So the stages are walked here instead, and every
boundary is hashed on the way past.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hashes, lora
from .song import Song, Step, seconds_to_tokens

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

    config = GenerationConfig(**song.generation_config)
    pipe = YuE2Pipeline(Path(models) / "YuE2-3B", Path(models) / "YuE2-Vae",
                        generation_config=config, progress=progress)
    adapters = lora.from_manifest(song.lora)
    if adapters:
        # Attached through the engine's hook rather than by patching pipe._model:
        # the engine rebuilds and moves that model between stages, so a patch
        # applied once from outside is silently dropped.
        pipe.on_model_ready.append(lora.hook(adapters))
    return pipe


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
                          "carried": carried_note(step)}
    return SongResult(audio=np.asarray(audio), sample_rate=SAMPLE_RATE, semantic=semantic,
                      latents=np.asarray(latents), config=config, weights=pipe.weights,
                      timing=timing,
                      request_identity=identity({"request": request.to_dict(), "config": config,
                                                 "weights": pipe.weights}))


def carried_note(step: Step) -> dict | None:
    return None if step.carry_from is None else {"from": step.carry_from, "tokens": step.carry_tokens}


def request_for(song: Song, step: Step):
    from yue2.protocol import SongRequest
    return SongRequest(style=song.style, lyrics=song.lyrics, id=f"{song.id}.{step.id}",
                       seed=step.seed, cot=song.cot, abc=song.abc, cfg_scale=song.cfg_scale)


def render_step(pipe, song: Song, step: Step, previous: Take | None = None) -> Take:
    """Render one step, carrying from `previous` when the step asks for it."""
    clock = time.perf_counter()
    stages: dict = {}
    carried: dict = {}

    plan = pipe.plan(request=request_for(song, step), abc_sampling=song.abc_sampling or None)
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
    semantic = pipe.generate_semantic(plan, sampling=sampling, carry=carry_tokens)
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
                              overlap_seconds=step.overlap_seconds)
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


def provenance(song: Song, takes: list[Take], models: Path) -> dict:
    """What a run needs to be reproducible later, resolved rather than referenced."""
    import torch
    from importlib.metadata import version

    return {
        "song": song.id,
        "rendered": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "yue2_infer": version("yue2-infer"),
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu",
            "models": str(models),
        },
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
