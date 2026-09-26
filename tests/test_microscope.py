"""The microscope's bookkeeping: what it records, and what it must leave alone."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from audiogen import microscope as scope
from audiogen import render
from audiogen import score as score_module

SCORE = """X:1
T:
M:4/4
L:1/16
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Am
% intro
V: Vocal
"Am"z16|"F"z16|
V: Ins
A4c4e4a4|Z|
% verse
V: Vocal
"Am"z4c2d2e4e4-|"F"e4z12|
V: Ins
Z2|
V: Vocal
"C"c2c2d2d2e8|"G"z16|
V: Ins
Z2|
"""
LYRICS = "[Verse]\nBlue light falls\nOn the kitchen wall\n"


def test_parse_reads_headers_sections_and_ties():
    s = score_module.parse(SCORE)
    assert (s.headers["Q"], s.headers["K"], s.sections) == ("1/4=120", "Am", ["intro", "verse"])
    vocal = [e for e in s.voices["Vocal"] if e.kind == "note"]
    # e4e4- then e4: the tied continuation is not a new note.
    assert [e.onset for e in vocal[:5]] == [True, True, True, True, False]
    assert s.bars["Vocal"] == 6
    assert s.seconds_per_unit() == pytest.approx(0.125)   # a 16th at 120 bpm


def test_parse_tolerates_truncation_mid_note():
    s = score_module.parse(SCORE[:SCORE.index("c2c2d2") + 3])
    assert s.sections == ["intro", "verse"]


def test_alignment_pairs_score_lines_with_lyric_lines():
    result = score_module.alignment(SCORE, LYRICS)
    verse = next(s for s in result["sections"] if s["label"] == "verse")
    assert verse["segmentation"] == "score line"
    first, second = verse["phrases"]
    assert (first["lyrics"], first["syllables"], first["melody_notes"]) == ("Blue light falls", 3, 4)
    assert first["notes_per_syllable"] == pytest.approx(4 / 3, abs=1e-3)
    assert first["bin"] == "1.3-1.6"
    assert (second["syllables"], second["melody_notes"]) == (5, 5)
    assert result["paired_phrases"] == 2


def test_unpairable_section_says_so_instead_of_guessing():
    result = score_module.alignment(SCORE, "[Verse]\nOne line only\nTwo\nThree\n")
    verse = next(s for s in result["sections"] if s["label"] == "verse")
    assert "not paired" in verse["phrase_pairing"]
    assert all(p["lyrics"] is None for p in verse["phrases"])


def test_development_marks_when_properties_arrive():
    cut = {n: SCORE[:n] for n in (20, 60, 200, len(SCORE))}
    report = score_module.development(cut)
    assert report["properties"]["key"]["final"] == "Am"
    assert report["properties"]["key"]["first_observable"] == 200
    assert report["properties"]["vocal_notes"]["share"][len(SCORE)] == 1.0


def test_valid_prefix_never_ends_mid_line():
    assert score_module.valid_prefix("X:1\nK:A\n\"Am\"c2d") == "X:1\nK:A\n"


def flow(step, state, chunk=0, steps=4, velocity=None):
    return SimpleNamespace(chunk_index=chunk, chunk_count=1, start=0, end=len(state), lead=0,
                           step=step, steps=steps, t=1.0 - step * (1.0 / steps), state=state,
                           velocity=velocity)


def test_recorder_copies_states_and_selects_checkpoints():
    recorder = scope.Recorder(ode_checkpoints=[0, 4])
    states = [torch.full((3, 64), float(i), dtype=torch.bfloat16) for i in range(5)]
    for i, s in enumerate(states):
        recorder.on_step(flow(i, s))
    got = recorder.states(0)
    assert sorted(got) == [0, 4]
    assert got[4].dtype == np.float32 and np.all(got[4] == 4.0)
    assert recorder.flow[0]["meta"]["dtype"] == "bfloat16"
    assert recorder.flow[0]["states"][4] is not states[4]          # copied, not held


def test_predicted_final_is_state_minus_t_times_velocity():
    recorder = scope.Recorder()
    recorder.on_step(flow(1, torch.full((2, 64), 3.0), velocity=torch.full((2, 64), 2.0)))
    recorder.on_step(flow(4, torch.full((2, 64), 1.5)))
    guess = recorder.predicted(0)
    assert np.all(guess[1] == 3.0 - 0.75 * 2.0)      # t = 0.75 at step 1 of 4
    assert np.all(guess[4] == 1.5)                    # the solution is its own estimate


def test_reference_capture_holds_the_solver_tensor():
    recorder = scope.Recorder(capture="reference")
    state = torch.zeros(2, 64)
    recorder.on_step(flow(0, state))
    assert recorder.flow[0]["states"][0].data_ptr() == state.data_ptr()


def test_recorder_token_streams_drop_only_the_end_token():
    recorder = scope.Recorder()
    for t in (5, 6, 7, 99):
        recorder.on_token("abc", t)
    assert recorder.stream("abc", end_token=99) == [5, 6, 7]
    assert recorder.stream("abc", end_token=42) == [5, 6, 7, 99]


def test_latent_metrics_measure_distance_to_the_solution():
    states = {0: np.ones((4, 64), np.float32), 1: np.zeros((4, 64), np.float32) + 0.5,
              2: np.zeros((4, 64), np.float32) + 0.25}
    rows = scope.latent_metrics(states)
    assert rows[-1]["delta_final"] == 0 and rows[-1]["cosine_final"] == pytest.approx(1)
    assert rows[0]["delta_previous"] is None and rows[1]["delta_previous"] == pytest.approx(16 * 0.5)


def test_audio_metrics_correlate_identical_audio_perfectly():
    t = np.arange(48000) / 48000
    audio = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    row = scope.audio_metrics(audio, audio, 48000)
    assert row["correlation_final"] == pytest.approx(1) and row["difference_rms_final"] == 0
    assert row["spectral_centroid_hz"] == pytest.approx(440, rel=0.05)
    assert row["peak"] == pytest.approx(1, abs=1e-3)


def test_listening_copy_is_separate_and_scaled():
    audio = np.full((10, 2), 0.1, np.float32)
    louder = scope.listening_copy(audio)
    assert audio.max() == pytest.approx(0.1) and louder.max() == pytest.approx(10 ** (-1 / 20))


def test_timeline_reports_first_step_judged_true(tmp_path):
    recorder = scope.Recorder()
    for i in range(3):
        recorder.on_step(flow(i, torch.zeros(2, 64), steps=2))
    notes = scope.annotation_template(recorder)
    notes["ode"]["chunk_000"]["1"]["vocal_present"] = False
    notes["ode"]["chunk_000"]["2"]["vocal_present"] = True
    scope.write_json(tmp_path / "analysis" / "annotations.json", notes)
    events = scope.timeline(tmp_path)
    assert events["ode"]["chunk_000"]["vocal_present"] == 2
    assert events["ode"]["chunk_000"]["words_intelligible"] is None


def test_checkpoints_adapt_to_the_stream_length():
    assert scope.checkpoints(640, scope.PLAN_CHECKPOINTS) == [50, 100, 200, 400, 640]
    assert scope.select_steps("all", 2) == {0, 1, 2}


def test_render_step_passes_observers_only_when_given(monkeypatch):
    calls = {}

    class Pipe:
        load_timing = {}

        def plan(self, **kw):
            calls["plan"] = kw
            return SimpleNamespace(abc_ids=[1], prefix=[1], abc="", timing={}, request=None)

        def generate_semantic(self, plan, **kw):
            calls["semantic"] = kw
            return SimpleNamespace(tokens=[1, 2], timing={})

        def synthesize(self, semantic, **kw):
            calls.setdefault("synth", []).append(kw)
            return np.zeros((2, 64), np.float32)

        def decode(self, latents):
            return np.zeros((10, 2), np.float32)

    monkeypatch.setattr(render, "request_for", lambda song, step: None)
    monkeypatch.setattr(render, "native_result", lambda *a: None)
    song = SimpleNamespace(abc_sampling={}, semantic_sampling={})
    step = SimpleNamespace(carry_from=None, new_tokens=2, blend_seconds=0, chunk_seconds=0,
                           overlap_seconds=0, id="take")
    render.render_step(Pipe(), song, step)
    assert "on_step" not in calls["synth"][0] and calls["plan"]["on_token"] is None
    observer = object()
    render.render_step(Pipe(), song, step, on_token=print, on_step=observer)
    assert calls["synth"][1]["on_step"] is observer and calls["semantic"]["on_token"] is print
    calls.pop("plan")
    held = SimpleNamespace(abc_ids=[9], prefix=[9], abc="", timing={}, request=None)
    take = render.render_step(Pipe(), song, step, plan=held)
    assert "plan" not in calls and take.stages["abc"]["tokens"] == 1


def test_band_correlation_separates_frequencies():
    t = np.arange(48000) / 48000
    low, high = np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 5000 * t)
    final = np.stack([low + high] * 2, axis=1)
    only_low = np.stack([low + np.sin(2 * np.pi * 5000 * t + 1.3)] * 2, axis=1)
    row = scope.band_correlations(only_low, final, 48000)
    assert row["sub/bass"] == pytest.approx(1, abs=1e-6)
    assert row["presence"] == pytest.approx(np.cos(1.3), abs=1e-3)


def test_envelope_similarity_ignores_phase():
    t = np.arange(96000) / 48000
    gate = (np.sin(2 * np.pi * 2 * t) > 0).astype(float)
    final = np.stack([gate * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    shifted = np.stack([gate * np.sin(2 * np.pi * 440 * t + 2.0)] * 2, axis=1)
    assert scope.audio_metrics(shifted, final, 48000)["correlation_final"] < 0
    assert scope.envelope_similarity(shifted, final, 48000)["envelope"] > 0.99


def test_envelope_drops_bands_with_no_frequency_bin():
    t = np.arange(48000) / 48000
    env = scope.envelope(np.sin(2 * np.pi * 440 * t), 48000)
    assert env.shape[1] == len(scope.envelope_centres(48000)) < 48
    freqs = np.fft.rfftfreq(4096, 1 / 48000)
    edges = np.geomspace(40, 16000, 49)
    assert len(scope.envelope_centres(48000)) == len(set(np.digitize(freqs[(freqs >= 40) & (freqs < 16000)], edges)))


def test_key_signature_applies_until_an_accidental_in_the_bar():
    s = score_module.parse("X:1\nL:1/8\nK:D#m\nV: Vocal\nF2 ^F2 =F2 F2 | F2 C2 G2 D2 B2 |\n")
    pitches = [e.pitch for e in s.voices["Vocal"]]
    # D#m has six sharps (F C G D A E); =F holds for the rest of its bar only.
    assert pitches[:4] == [66, 66, 65, 65]
    assert pitches[4:] == [66, 61, 68, 63, 71]
    assert score_module.key_signature("Bb") == {"B": -1, "E": -1}
    assert score_module.key_signature("Dorian nonsense") == {}


def test_a_line_holding_only_a_tied_tail_joins_the_previous_phrase():
    text = SCORE.replace('"C"c2c2d2d2e8|"G"z16|', '"C"c2c2d2d2e8-|\nV: Vocal\n"G"e4z12|')
    s = score_module.parse(text)
    verse = [e for e in s.voices["Vocal"] if e.section == 1]
    groups = score_module.by_line(verse)
    assert all(any(e.kind == "note" and e.onset for e in g) for g in groups)
