"""ComfyUI's launch flags, accepted verbatim.

The album ran behind three of them and the command line was the one thing
nobody wrote down. Taking the same strings means the two stacks are started the
same way and argv records it.
"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("render_cli", Path(__file__).parent.parent / "bin" / "render.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def args(**kw):
    import argparse
    base = dict(use_pytorch_cross_attention=False, disable_smart_memory=False, reserve_vram=None)
    return argparse.Namespace(**{**base, **kw})


class TestFlagMapping:
    def test_no_flags_means_no_overrides(self):
        assert cli.comfyui_parity(args()) == {}

    def test_cross_attention_selects_the_sdpa_backend(self):
        assert cli.comfyui_parity(args(use_pytorch_cross_attention=True)) == {"backend": "torch"}

    def test_disable_smart_memory_offloads_idle_modules(self):
        assert cli.comfyui_parity(args(disable_smart_memory=True)) == {"offload_ar": True}

    def test_reserve_vram_is_subtracted_from_physical_memory(self):
        budget = cli.comfyui_parity(args(reserve_vram=8.0))["memory_budget_gib"]
        import os
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        assert budget == pytest.approx(total - 8.0)

    def test_reserving_more_than_exists_is_refused(self):
        with pytest.raises(SystemExit, match="leaves nothing"):
            cli.comfyui_parity(args(reserve_vram=100000.0))

    def test_the_albums_whole_flag_line_maps(self):
        got = cli.comfyui_parity(args(use_pytorch_cross_attention=True,
                                      disable_smart_memory=True, reserve_vram=8.0))
        assert set(got) == {"backend", "offload_ar", "memory_budget_gib"}


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
