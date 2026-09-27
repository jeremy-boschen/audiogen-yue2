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
