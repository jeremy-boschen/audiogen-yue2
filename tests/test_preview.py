"""Listen-so-far: previews of the tokens written so far, taken mid-stage.

The bit-identity of the finished take was checked on real models (DONE.md,
2026-09-26: Sunday Kitchen s2026 with previews at 10 s and 30 s matched the plain
render and the original capture at every stage). These tests hold the logic.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from yue2.protocol import CODEC_OFFSET, MUSIC_END

from audiogen import render


class Pipe:
    load_timing = {}

    def __init__(self, quantization="none", emit=(5, 6, 7, 8)):
        self.quantization, self.emit, self.events = quantization, emit, []

    def plan(self, **kw):
        return SimpleNamespace(abc_ids=[1], prefix=[1], abc="", timing={}, request=None)

    def generate_semantic(self, plan, *, on_token=None, carry=None, **kw):
        for value in self.emit:
            self.events.append(("token", value))
            if on_token is not None:
                on_token("semantic", value + CODEC_OFFSET)
        if on_token is not None:
            on_token("semantic", MUSIC_END)          # the end token is not audio
        return SimpleNamespace(tokens=list(carry or []) + list(self.emit), timing={})

    def synthesize(self, semantic, **kw):
        self.events.append(("synthesize", list(semantic.tokens)))
        return np.zeros((len(semantic.tokens), 64), np.float32)

    def decode(self, latents):
        self.events.append(("decode", len(latents)))
        return np.zeros((len(latents) * 1920, 2), np.float32)

    def _stage_boundary(self):
        self.events.append(("boundary",))

    def _load_model(self, for_nar=False):
        self.events.append(("load", for_nar))


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(render, "request_for", lambda song, step: None)
    monkeypatch.setattr(render, "native_result", lambda *a: None)
    song = SimpleNamespace(abc_sampling={}, semantic_sampling={})
    step = SimpleNamespace(carry_from=None, new_tokens=4, blend_seconds=0, chunk_seconds=0,
                           overlap_seconds=0, id="take")
    return song, step


def test_a_preview_voices_the_tokens_so_far_then_hands_the_model_back(plain):
    song, step = plain
    pipe, heard = Pipe(), []
    asks = iter([False, True])
    preview = render.Preview(lambda: next(asks), lambda audio, seconds: heard.append((len(audio), seconds)), every=2)
    render.render_step(pipe, song, step, preview=preview)
    # asked after token 2 (no) and token 4 (yes): voices all four, then re-arms the AR stage
    i = pipe.events.index(("synthesize", [5, 6, 7, 8]))
    assert pipe.events[i - 1] == ("token", 8)
    assert pipe.events[i + 1:i + 4] == [("decode", 4), ("boundary",), ("load", False)]
    assert heard == [(4 * 1920, 4 / 25)]
    assert pipe.events[-2:] == [("synthesize", [5, 6, 7, 8]), ("decode", 4)]   # the take itself


def test_a_request_before_the_minimum_waits_for_it(plain):
    song, step = plain
    pipe, asked = Pipe(), []
    preview = render.Preview(lambda: asked.append(1) or True, lambda audio, seconds: None, every=1, min_tokens=3)
    render.render_step(pipe, song, step, preview=preview)
    first = next(e for e in pipe.events if e[0] == "synthesize")
    assert first == ("synthesize", [5, 6, 7])          # not polled at tokens 1 and 2


def test_no_preview_is_taken_unless_asked(plain):
    song, step = plain
    pipe = Pipe()
    render.render_step(pipe, song, step, preview=render.Preview(lambda: False, None, every=1))
    assert [e for e in pipe.events if e[0] == "synthesize"] == [("synthesize", [5, 6, 7, 8])]


def test_the_observer_still_sees_every_token(plain):
    song, step = plain
    seen = []
    render.render_step(Pipe(), song, step, on_token=lambda phase, token: seen.append(token),
                       preview=render.Preview(lambda: False, None))
    assert seen == [v + CODEC_OFFSET for v in (5, 6, 7, 8)] + [MUSIC_END]


def test_fp8_refuses_previews(plain):
    song, step = plain
    with pytest.raises(ValueError, match="quantization"):
        render.render_step(Pipe(quantization="fp8"), song, step, preview=render.Preview(lambda: True, None))
