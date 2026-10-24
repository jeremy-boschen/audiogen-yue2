"""CLI overrides and manifest engine configuration."""
import argparse
import importlib.util
from pathlib import Path
import pytest
spec = importlib.util.spec_from_file_location('render_cli', Path(__file__).parent.parent / 'bin/render.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

def test_only_explicit_options_override_manifest():
    args = argparse.Namespace(backend=None, device=None, offload_ar=None, memory_budget_gib=None)
    assert cli.pipeline_options(args) == {}
    args.offload_ar = False
    args.memory_budget_gib = 24
    assert cli.pipeline_options(args) == {'offload_ar': False, 'memory_budget_gib': 24}

class TestPipelineKeys:
    def test_a_typo_in_a_pipeline_key_is_refused(self, tmp_path):
        import json
        from audiogen import render, song as song_module
        (tmp_path / "style.txt").write_text("rock")
        (tmp_path / "lyrics.txt").write_text("[verse]\nwords")
        (tmp_path / "song.json").write_text(json.dumps(
            {"id": "s", "seed": 1, "pipeline": {"memory_budget_gb": 56},
             "steps": [{"id": "a", "seconds": 10.0}]}))
        song = song_module.load(tmp_path)
        with pytest.raises(ValueError, match="unknown pipeline keys"):
            render.build_pipeline(tmp_path, song)

    def test_the_manifest_carries_pipeline_settings_through(self, tmp_path):
        import json
        from audiogen import song as song_module
        (tmp_path / "style.txt").write_text("rock")
        (tmp_path / "lyrics.txt").write_text("[verse]\nwords")
        (tmp_path / "song.json").write_text(json.dumps(
            {"id": "s", "seed": 1, "pipeline": {"offload_ar": True},
             "steps": [{"id": "a", "seconds": 10.0}]}))
        assert song_module.load(tmp_path).pipeline == {"offload_ar": True}


class TestFlashFallback:
    """An explicit "torch" backend must not reach the CUDA-graph path on a torch built without
    FlashAttention (PyTorch's Windows wheels): it raised "USE_FLASH_ATTENTION was not enabled"."""

    def build(self, tmp_path, monkeypatch, flash_built, backend):
        import json
        import yue2.pipeline
        from audiogen import render, song as song_module
        seen = {}

        class FakePipeline:
            def __init__(self, *args, **options):
                seen.update(options)
                self.on_model_ready = []

        monkeypatch.setattr(yue2.pipeline, "YuE2Pipeline", FakePipeline)
        monkeypatch.setattr(render, "flash_attention_built", lambda torch: flash_built)
        (tmp_path / "style.txt").write_text("rock")
        (tmp_path / "lyrics.txt").write_text("[verse]\nwords")
        (tmp_path / "song.json").write_text(json.dumps(
            {"id": "s", "seed": 1, "pipeline": {"backend": backend}, "steps": [{"id": "a", "seconds": 10.0}]}))
        render.build_pipeline(tmp_path, song_module.load(tmp_path))
        return seen["backend"]

    def test_should_fall_back_to_torch_eager_when_torch_is_requested_without_flash(self, tmp_path, monkeypatch):
        assert self.build(tmp_path, monkeypatch, flash_built=False, backend="torch") == "torch-eager"

    def test_should_keep_torch_when_flash_is_built(self, tmp_path, monkeypatch):
        assert self.build(tmp_path, monkeypatch, flash_built=True, backend="torch") == "torch"


class TestContinuationProfiles:
    """A carry chain is refused before anything renders under a profile that cannot continue,
    and allowed under 'official-continuation' on any machine; rng_device is coerced either way."""

    def song(self, tmp_path, rng="cpu"):
        import json
        from audiogen import song as song_module
        (tmp_path / "style.txt").write_text("rock")
        (tmp_path / "lyrics.txt").write_text("[verse]\nwords")
        (tmp_path / "song.json").write_text(json.dumps(
            {"id": "s", "seed": 1, "generation_config": {"rng_device": rng},
             "steps": [{"id": "a", "seconds": 10.0},
                       {"id": "b", "seconds": 20.0, "carry_from": "a", "carry_seconds": 5.0}]}))
        return song_module.load(tmp_path)

    def test_official_refuses_and_names_the_profile_that_can(self, tmp_path):
        from audiogen import render
        with pytest.raises(ValueError, match="'official-continuation'"):
            render.fit_generation_config(self.song(tmp_path), "official")

    def test_official_continuation_continues_with_auto_rng(self, tmp_path, capsys):
        from audiogen import render
        config = render.fit_generation_config(self.song(tmp_path), "official-continuation")
        assert config["rng_device"] == "auto"
        assert "not available" in capsys.readouterr().out

    def test_the_mps_profile_keeps_its_cpu_rng(self, tmp_path):
        from audiogen import render
        assert render.fit_generation_config(self.song(tmp_path), "comfyui-yue2-mps-v1")["rng_device"] == "cpu"
