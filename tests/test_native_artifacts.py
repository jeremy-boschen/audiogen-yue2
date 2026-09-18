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


class TestProvenanceRecordsTheInvocation:
    """The ComfyUI launch flags changed the audio and lived only in a transcript.

    Anything that decides what gets rendered has to land in the record, not in
    the shell history of whoever happened to run it.
    """

    def a_song(self, tmp_path):
        import json as _json
        from audiogen import song as song_module
        (tmp_path / "style.txt").write_text("rock, female vocal")
        (tmp_path / "lyrics.txt").write_text("[verse]\nsome words")
        (tmp_path / "song.json").write_text(_json.dumps(
            {"id": "s", "seed": 1, "steps": [{"id": "a", "seconds": 10.0}]}))
        return song_module.load(tmp_path)

    def test_the_command_line_is_recorded(self, tmp_path, monkeypatch):
        from audiogen import render
        monkeypatch.setattr(sys, "argv", ["bin/render.py", "burn_it_down", "--lora", "x.safetensors:nar"])
        record = render.provenance(self.a_song(tmp_path), [], tmp_path)
        assert record["invocation"]["argv"][-1] == "x.safetensors:nar"
        assert record["invocation"]["executable"] == sys.executable

    def test_the_interpreter_is_recorded_not_just_torch(self, tmp_path):
        from audiogen import render
        record = render.provenance(self.a_song(tmp_path), [], tmp_path)
        assert record["environment"]["python"] == ".".join(str(n) for n in sys.version_info[:3])
        assert record["environment"]["torch"]

    def test_numeric_env_vars_are_captured_and_others_left_alone(self, tmp_path, monkeypatch):
        from audiogen import render
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        monkeypatch.setenv("SOME_UNRELATED_SECRETISH_VAR", "nope")
        env = render.provenance(self.a_song(tmp_path), [], tmp_path)["invocation"]["env"]
        assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert "SOME_UNRELATED_SECRETISH_VAR" not in env


class TestProvenanceNeverCopiesACredential:
    """provenance.json travels with the audio into every output directory."""

    def test_a_token_is_named_but_not_copied(self, monkeypatch):
        from audiogen import render
        monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 34)
        monkeypatch.setenv("HF_HOME", "/tmp/hf")
        env = render.recorded_env()
        assert env["HF_TOKEN"] == "<set, not recorded>"
        assert "a" * 34 not in json.dumps(env)
        assert env["HF_HOME"] == "/tmp/hf", "non-secret settings still recorded"

    def test_every_secretish_spelling_is_covered(self, monkeypatch):
        from audiogen import render
        for name in ("HF_TOKEN", "CUDA_API_KEY", "TORCH_AUTH", "MKL_SECRET_X"):
            monkeypatch.setenv(name, "sensitive-value")
        env = render.recorded_env()
        assert "sensitive-value" not in json.dumps(env)
