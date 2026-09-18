"""A song is a directory, and song.json is its contract.

A step names a take. Steps that carry from an earlier step are how a track gets
built: generate a short idea, then grow it, then replace its tail, each time
keeping the audio already decided.

    {
      "id": "burn_it_down",
      "seed": 777,
      "cot": "full",
      "semantic_sampling": {"temperature": 1.0, "max_tokens": 5000},
      "steps": [
        {"id": "riff",   "seconds": 45.0},
        {"id": "grown",  "seconds": 201.2, "carry_from": "riff",  "carry_seconds": 45.0},
        {"id": "tail",   "seconds": 201.1, "carry_from": "grown", "carry_seconds": 180.0, "seed": 779}
      ]
    }

``carry_seconds`` is how much of the earlier take is kept, measured from its
start. It is always explicit: a sentinel meaning "all of it" reads fine until the
take length changes underneath it.

Seconds are a convenience over what the engine actually takes, and a convenience
that cannot be bypassed is a cage. Every derived value has an escape hatch that
names the engine's own unit and wins outright:

    seconds       -> max_tokens     "max_tokens": 3875
    carry_seconds -> carry tokens   "carry_tokens": 1125

Both forms may be given for one quantity -- a manifest is allowed to spell out
what it means -- but they must agree. A mismatch is an error naming both values,
never a silent precedence rule, because a precedence rule is only discovered once
the two have already drifted apart.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

TOKENS_PER_SECOND = 25


def seconds_to_tokens(seconds: float) -> int:
    return round(float(seconds) * TOKENS_PER_SECOND)


@dataclass
class Step:
    id: str
    seed: int
    seconds: float | None = None
    max_tokens: int | None = None          # engine unit; wins over `seconds`
    carry_from: str | None = None
    carry_seconds: float | None = None
    carry_token_count: int | None = None   # engine unit; wins over `carry_seconds`
    blend_seconds: float = 0.0
    chunk_seconds: float = 0.0
    overlap_seconds: float = 0.0

    @property
    def carry_tokens(self) -> int:
        if self.carry_token_count is not None:
            return self.carry_token_count
        return seconds_to_tokens(self.carry_seconds or 0.0)

    @property
    def new_tokens(self) -> int:
        """The budget for NEW tokens, which is what the engine's max_tokens means.

        Set `max_tokens` and it is passed through untouched -- that is the engine's
        own unit and its own meaning. Set `seconds` and the carry is subtracted
        here, so a manifest can state the length the take should end up at, which
        is the thing a person means.
        """
        if self.max_tokens is not None:
            return self.max_tokens
        return seconds_to_tokens(self.seconds) - self.carry_tokens

    @property
    def total_tokens(self) -> int:
        """What the take should end up as, however the step was specified."""
        return self.new_tokens + self.carry_tokens

    @property
    def target_seconds(self) -> float:
        return round(self.total_tokens / TOKENS_PER_SECOND, 2)


@dataclass
class Song:
    root: Path
    id: str
    seed: int
    cot: str = "full"
    cfg_scale: float | None = None
    generation_config: dict = field(default_factory=dict)
    abc_sampling: dict = field(default_factory=dict)
    semantic_sampling: dict = field(default_factory=dict)
    lora: list[dict] = field(default_factory=list)
    pipeline: dict = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)

    @property
    def style(self) -> str:
        return (self.root / "style.txt").read_text().strip()

    @property
    def lyrics(self) -> str:
        return (self.root / "lyrics.txt").read_text().strip()

    @property
    def abc(self) -> str | None:
        path = self.root / "score.abc"
        return path.read_text() if path.exists() else None

    def step(self, step_id: str) -> Step:
        for candidate in self.steps:
            if candidate.id == step_id:
                return candidate
        raise KeyError(f"no step {step_id!r} in {self.id}")


def load(root: Path) -> Song:
    root = Path(root)
    spec = json.loads((root / "song.json").read_text())
    seed = spec["seed"]
    known = {"id", "seed", "seconds", "max_tokens", "carry_from", "carry_seconds",
             "carry_tokens", "blend_seconds", "chunk_seconds", "overlap_seconds"}
    steps = []
    for raw in spec["steps"]:
        unknown = set(raw) - known
        if unknown:
            # A typo in a manifest key would otherwise be silently ignored and the
            # step would render with a default nobody asked for.
            raise ValueError(f"step {raw.get('id')!r}: unknown keys {sorted(unknown)}")
        steps.append(Step(id=raw["id"], seed=raw.get("seed", seed),
                          seconds=raw.get("seconds"), max_tokens=raw.get("max_tokens"),
                          carry_from=raw.get("carry_from"),
                          carry_seconds=raw.get("carry_seconds"),
                          carry_token_count=raw.get("carry_tokens"),
                          blend_seconds=raw.get("blend_seconds", 0.0),
                          chunk_seconds=raw.get("chunk_seconds", 0.0),
                          overlap_seconds=raw.get("overlap_seconds", 0.0)))
    song = Song(root=root, id=spec.get("id", root.name), seed=seed, cot=spec.get("cot", "full"),
                cfg_scale=spec.get("cfg_scale"), generation_config=spec.get("generation_config", {}),
                abc_sampling=spec.get("abc_sampling", {}),
                semantic_sampling=spec.get("semantic_sampling", {}),
                lora=spec.get("lora", []), pipeline=spec.get("pipeline", {}), steps=steps)
    validate(song)
    return song


# How the engine is constructed, as opposed to how it samples. Held apart from
# generation_config because these decide memory layout and tiling, and a wrong
# key here would otherwise be swallowed as a sampling override nobody asked for.
PIPELINE_KEYS = {"backend", "quantization", "memory_budget_gib", "vae_core_frames",
                 "offload_ar", "device"}


def validate(song: Song) -> None:
    """Fail on a manifest that cannot render, before any weights are loaded."""
    for required in ("style.txt", "lyrics.txt"):
        if not (song.root / required).exists():
            raise FileNotFoundError(song.root / required)
    if not song.steps:
        raise ValueError(f"{song.id}: no steps")

    for entry in song.lora:
        unknown = set(entry) - {"path", "branch", "strength"}
        if unknown:
            raise ValueError(f"{song.id}: lora entry has unknown keys {sorted(unknown)}")
        if "path" not in entry or "branch" not in entry:
            raise ValueError(f"{song.id}: a lora entry needs both path and branch")

    seen: set[str] = set()
    for step in song.steps:
        where = f"{song.id}: step {step.id!r}"
        if step.id in seen:
            raise ValueError(f"{song.id}: duplicate step id {step.id!r}")

        if step.seconds is None and step.max_tokens is None:
            raise ValueError(f"{where} sets neither seconds nor max_tokens")

        # A convenience and its escape hatch may both be given -- a manifest is
        # allowed to spell out what it means -- but a mismatch is never resolved by
        # precedence. Carry is checked first: the length check derives from it.
        if step.carry_seconds is not None and step.carry_token_count is not None:
            derived = seconds_to_tokens(step.carry_seconds)
            if derived != step.carry_token_count:
                raise ValueError(
                    f"{where} disagrees with itself: carry_seconds={step.carry_seconds} is "
                    f"{derived} tokens, but carry_tokens={step.carry_token_count} was given")
        if step.seconds is not None and step.max_tokens is not None:
            derived = seconds_to_tokens(step.seconds) - step.carry_tokens
            if derived != step.max_tokens:
                raise ValueError(
                    f"{where} disagrees with itself: seconds={step.seconds} less "
                    f"{step.carry_tokens} carried is {derived} new tokens, but "
                    f"max_tokens={step.max_tokens} was given")

        if step.carry_from is not None:
            if step.carry_from not in seen:
                raise ValueError(f"{where} carries from {step.carry_from!r}, "
                                 "which is not an earlier step")
            source = song.step(step.carry_from)
            if step.carry_tokens <= 0:
                raise ValueError(f"{where} sets carry_from but keeps nothing; "
                                 "say how much with carry_seconds or carry_tokens")
            if step.carry_tokens > source.total_tokens:
                raise ValueError(f"{where} keeps {step.carry_tokens} tokens of "
                                 f"{step.carry_from!r}, which is only {source.total_tokens} long")
        elif step.carry_seconds is not None or step.carry_token_count is not None:
            raise ValueError(f"{where} says what to carry but not carry_from")

        if step.new_tokens < 1:
            raise ValueError(f"{where} keeps {step.carry_tokens} tokens of a "
                             f"{step.total_tokens}-token take, leaving nothing to generate")
        seen.add(step.id)
