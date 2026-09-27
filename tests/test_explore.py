"""Data prep for the explorers: small synthetic inputs, no model, no real runs."""
import base64
import json

import numpy as np
import pytest

from audiogen import explore

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
z16|
V: Ins
A,4B,4C4D4|
% verse
V: Vocal
c4d4-d4e4|
V: Ins
Z|
"""


def test_pool_columns_averages_and_upsamples():
    grid = np.arange(12, dtype=float).reshape(6, 2)
    pooled = explore.pool_columns(grid, 3)
    assert pooled.shape == (3, 2)
    assert np.allclose(pooled[:, 0], [1, 5, 9])
    assert explore.pool_columns(grid[:2], 4).shape == (4, 2)


def test_quantize_clips_to_bytes():
    q = explore.quantize(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), 0.0, 1.0)
    assert q.dtype == np.uint8
    assert q.tolist() == [0, 0, 128, 255, 255]


def test_band_correlation_per_band_and_flat_bands():
    t = np.linspace(0, 6, 50)
    final = np.stack([np.sin(t), np.cos(t), np.zeros_like(t)], axis=1)
    grid = np.stack([np.sin(t), -np.cos(t), np.ones_like(t)], axis=1)
    r = explore.band_correlation(grid, final)
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(-1.0)
    assert r[2] == 0.0            # no variation: nothing to agree about


def test_settled_from_requires_staying_above():
    assert explore.settled_from([0.1, 0.95, 0.5, 0.92, 0.99], 0.9) == 3
    assert explore.settled_from([0.1, 0.2], 0.9) is None


def test_occupied_bands_drops_bands_without_fft_bins():
    keep = explore.occupied_bands(48)
    assert 0 not in keep and len(keep) < 48 and keep[-1] == 47


def test_project_takes_places_near_duplicates_together():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    features = np.stack([a, a + 0.01 * rng.normal(size=200), b, b + 0.01 * rng.normal(size=200)])
    coords, corr = explore.project_takes(features)
    assert coords.shape == (4, 3)
    assert corr[0, 1] > 0.99 and abs(corr[0, 2]) < 0.3
    near = np.linalg.norm(coords[0] - coords[1])
    far = np.linalg.norm(coords[0] - coords[2])
    assert near < far


def test_classical_mds_recovers_a_line():
    x = np.array([0.0, 1.0, 3.0])
    d = np.abs(x[:, None] - x[None, :])
    coords = explore.classical_mds(d, 3)
    rebuilt = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    assert np.allclose(rebuilt, d, atol=1e-9)


def test_score_layer_times_and_ties():
    layer = explore.score_layer(SCORE)
    spu = layer["seconds_per_unit"]
    assert spu == pytest.approx(0.125)                 # 1/16 at quarter = 120
    vocal = layer["voices"]["Vocal"]
    assert vocal["t0"] == [2.0, 2.5, 3.5]               # the tied d is one sounding note
    assert vocal["dur"] == [0.5, 1.0, 0.5]
    assert [s["label"] for s in layer["sections"]] == ["intro", "verse"]
    assert layer["sections"][1]["start"] == pytest.approx(2.0)
    assert layer["voices"]["Ins"]["seconds"] == pytest.approx(4.0)


def test_plan_reveal_counts_notes_written_by_each_prefix():
    final = explore.score_layer(SCORE)["voices"]
    cut = SCORE.index("c4d4")
    reveal = explore.plan_reveal({10: SCORE[:cut], 20: SCORE}, final)
    assert reveal[0]["notes"] == {"Vocal": 0, "Ins": 4}
    assert reveal[1]["notes"] == {"Vocal": 3, "Ins": 4}


def test_dedupe_takes_merges_identical_pcm():
    takes = [{"run": "a", "group": "baseline", "pcm_hash": "x"}, {"run": "b", "group": "seed study", "pcm_hash": "x"},
             {"run": "c", "group": "seed study", "pcm_hash": "y"}, {"run": "d", "group": "noise seed", "pcm_hash": None}]
    kept = explore.dedupe_takes(takes)
    assert [t["run"] for t in kept] == ["a", "c", "d"]
    assert kept[0]["aliases"] == ["b (seed study)"]


def test_discover_runs_only_finished(tmp_path):
    done = tmp_path / "study.seed.s1"
    busy = tmp_path / "study.seed.s2"
    for run, meta in ((done, {"finished": "2026-01-01T00:00:00"}), (busy, {"started": "now"})):
        run.mkdir()
        (run / "metadata.json").write_text(json.dumps(meta))
    assert explore.discover_runs([tmp_path]) == [done.resolve()]


def test_takes_in_run_groups_and_labels(tmp_path):
    import soundfile as sf
    run = tmp_path / "burn.fixed_abc.s42"
    (run / "final").mkdir(parents=True)
    sf.write(run / "final/audio.wav", np.zeros((10, 2), dtype=np.float32), 48000)
    (run / "analysis").mkdir()
    (run / "analysis/alignment.json").write_text(json.dumps({"song": {"notes_per_syllable": 1.2}}))
    takes = explore.takes_in_run(run)
    assert len(takes) == 1
    assert takes[0]["group"] == "fixed ABC" and takes[0]["seed"] == 42 and takes[0]["notes_per_syllable"] == 1.2


def test_js_global_is_a_script_assignment():
    text = explore.js_global("FOO", {"a": [1, 2]})
    assert text.startswith("window.FOO = ") and text.rstrip().endswith(";")
    assert json.loads(text[len("window.FOO = "):].rstrip().rstrip(";")) == {"a": [1, 2]}
    assert base64.b64decode(explore.b64(np.array([1, 2], dtype=np.uint8))) == b"\x01\x02"


def test_envelope_grid_shape(tmp_path):
    import soundfile as sf
    rate = 48000
    t = np.arange(rate * 2) / rate
    sf.write(tmp_path / "a.wav", np.sin(2 * np.pi * 440 * t).astype(np.float32), rate)
    grid = explore.envelope_grid(tmp_path / "a.wav", 10, 16)
    keep = explore.occupied_bands(16)
    assert grid.shape == (10, len(keep))
    loudest = keep[grid.max(axis=0).argmax()]
    assert loudest == int(np.argmax(np.array(explore.band_edges(16)) > 440)) - 1


# --- the stack's phrase list: pickups folded into the phrase they lead into ---

def pickup_phrase(section, start, seconds):
    return {"section_index": section, "start": start, "seconds": seconds}


def test_a_note_ending_a_section_that_runs_into_the_next_is_its_pickup():
    # Slow Down take 2: the chorus's last "phrase" is one 0.17 s note, the verse's "My"
    out = explore.fold_pickups([pickup_phrase(1, 42.95, 1.36), pickup_phrase(1, 46.19, 0.17), pickup_phrase(2, 46.36, 5.45)])
    assert [(p["start"], p["seconds"], p.get("pickup")) for p in out] == [(42.95, 1.36, None), (46.19, 5.62, True)]


def test_a_short_phrase_inside_a_section_or_before_a_rest_stays():
    kept = [pickup_phrase(1, 10.0, 0.5), pickup_phrase(1, 10.5, 2.0), pickup_phrase(1, 13.0, 0.4), pickup_phrase(2, 14.0, 2.0)]
    assert explore.fold_pickups(kept) == kept
