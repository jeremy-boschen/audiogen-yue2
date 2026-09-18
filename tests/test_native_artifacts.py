"""A take has to write what the skill's listening page will accept.

listen.py re-hashes every file it copies against result.json and withholds the
player when they disagree, so the only honest test is to run it.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from audiogen.render import SAMPLE_RATE, Take

LISTEN = Path.home() / "dev/projects/YuE/skills/yue2-music/scripts/listen.py"


def a_take(step_id="a", seed=1234):
    """A SongResult small enough to write in milliseconds, real enough to save."""
    from yue2.pipeline import SemanticResult, SongResult, SymbolicPlan
    from yue2.protocol import SongRequest

    request = SongRequest(style="smoky low female vocal, moody guitar rock",
                          lyrics="[verse]\nthe lights went down\n", id=f"t.{step_id}", seed=seed)
    plan = SymbolicPlan(request, "X:1\nT:t\nK:F\n", [1, 2, 3], [4, 5, 6])
    semantic = SemanticResult(plan, [7, 8, 9], {}, False)
    seconds = 0.25
    audio = np.zeros((int(SAMPLE_RATE * seconds), 2), dtype=np.float32)
    result = SongResult(audio=audio, sample_rate=SAMPLE_RATE, semantic=semantic,
                        latents=np.zeros((int(seconds * 25), 64), dtype=np.float32),
                        config={"decoder_release": "test", "validation_status": "unvalidated",
                                "audiogen": {"step": step_id, "adapters": None, "carried": None}},
                        weights={"mot": "deadbeef"}, timing={}, request_identity=f"id-{seed}")
    return Take(step_id=step_id, semantic=list(semantic.tokens), latents=result.latents,
                audio=audio, score=plan.abc, stages={}, seconds=seconds,
                render_seconds=0.0, result=result)


def test_write_lays_down_the_native_receipt(tmp_path):
    a_take().write(tmp_path)
    for name in ("audio.flac", "request.json", "config.json", "result.json",
                 "score.abc", "semantic.npy", "latent.npy"):
        assert (tmp_path / name).is_file(), f"{name} missing"
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "complete"
    # Nothing unreceipted: every file but the receipt itself is in the receipt.
    written = {p.name for p in tmp_path.iterdir()} - {"result.json"}
    assert written == set(result["artifacts"]), written ^ set(result["artifacts"])
    assert result["artifacts"]["audio.flac"]["bytes"] == (tmp_path / "audio.flac").stat().st_size


def test_the_receipt_is_the_producers_not_the_directorys(tmp_path):
    """Rewriting the audio after the fact must break the hash, or it proves nothing."""
    a_take().write(tmp_path)
    audio = tmp_path / "audio.flac"
    audio.write_bytes(audio.read_bytes()[:-64])
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["artifacts"]["audio.flac"]["bytes"] != audio.stat().st_size


@pytest.mark.skipif(not LISTEN.is_file(), reason="engine skill not checked out")
def test_listen_page_accepts_two_takes(tmp_path):
    for step, seed in (("a", 1234), ("b", 5678)):
        a_take(step, seed).write(tmp_path / step)
    out = tmp_path / "compare"
    done = subprocess.run([sys.executable, str(LISTEN), str(tmp_path / "a"), str(tmp_path / "b"),
                           "--output", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, f"listen.py flagged the bundle:\n{done.stdout}\n{done.stderr}"
    page = (out / "index.html").read_text()
    assert page.count("<audio controls") == 2, "a player was withheld"
    manifest = json.loads((out / "manifest.json").read_text())
    assert [c["status"] for c in manifest["cases"]] == ["complete", "complete"]
    # Each case's audio passed the native hash check, not merely got copied.
    assert all(c["native_artifact_checks"]["audio.flac"] == "passed" for c in manifest["cases"])
