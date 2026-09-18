"""Measured regression: stock MPS RMSNorm differs on this captured BF16 row."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from audiogen.vendor import metal_norm


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS numerical reference')
def test_metal_norm_matches_captured_reference_bits():
    root = Path(__file__).parent / 'fixtures'
    fixture = np.load(root / 'metal_norm_row.npz')
    x, weight, expected = [torch.tensor(fixture[k], dtype=torch.bfloat16, device='mps')
                           for k in ('x', 'weight', 'expected')]
    epsilon = json.loads((root / 'metal_norm_row.json').read_text())['epsilon']
    with torch.inference_mode():
        actual = metal_norm.fused_rmsnorm_modulate(x.view(-1, x.shape[-1]), weight, epsilon).view_as(x)
        stock = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, epsilon)
    assert metal_norm._last_backend == 'kernel'
    assert torch.equal(actual.cpu().view(torch.uint8), expected.cpu().view(torch.uint8))
    assert not torch.equal(stock.cpu().view(torch.uint8), expected.cpu().view(torch.uint8))
