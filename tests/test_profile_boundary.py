"""Production cannot label a globally patched engine as an official profile."""
from pathlib import Path

import pytest

from audiogen import render


def test_production_rejects_active_diagnostic_shims_before_loading(monkeypatch):
    monkeypatch.setattr(render.shims, 'active', lambda: ['rmsnorm'])
    with pytest.raises(ValueError, match='restart without experimental shims'):
        render.build_pipeline(Path('/not-loaded'), None)
