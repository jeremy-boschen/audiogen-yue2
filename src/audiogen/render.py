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

from . import hashes
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

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        (directory / "score.abc").write_text(self.score)
        np.save(directory / "semantic.npy", np.asarray(self.semantic, dtype=np.int32))
        np.save(directory / "latent.npy", np.asarray(self.latents, dtype=np.float32))
        sf.write(directory / f"{self.step_id}.flac", self.audio, SAMPLE_RATE, subtype="PCM_24")
        return directory


def build_pipeline(models: Path, song: Song, *, progress: bool = True):
    from yue2.pipeline import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    config = GenerationConfig(**song.generation_config)
    return YuE2Pipeline(Path(models) / "YuE2-3B", Path(models) / "YuE2-Vae",
                        generation_config=config, progress=progress)


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

    return Take(step_id=step.id, semantic=list(semantic.tokens), latents=np.asarray(latents),
                audio=np.asarray(audio), score=plan.abc, stages=stages,
                seconds=round(len(semantic.tokens) / 25, 2),
                render_seconds=round(time.perf_counter() - clock, 1),
                plan_prefix_tokens=len(plan.prefix), carried=carried)


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
