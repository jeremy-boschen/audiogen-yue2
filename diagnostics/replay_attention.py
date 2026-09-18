"""Replay decode attention on the same captured Q/K/V with original strides."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from yue2.profiles import ComfyUIYuE2MPSProfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('trace', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    tensors = torch.load(args.trace / 'tensors.pt', map_location='cpu', weights_only=True)
    metadata = json.loads((args.trace / 'trace.json').read_text())
    base = '0001/model.layers.0.self_attn/attention_'
    profile = ComfyUIYuE2MPSProfile()
    from metal_flash_attn import sdpa as mfa

    def restore(name):
        original = metadata[base + name]
        tensor = tensors[base + name]
        result = torch.empty_strided(original['shape'], original['stride'], dtype=tensor.dtype, device='mps')
        result.copy_(tensor)
        return result.transpose(1, 2)

    with torch.inference_mode():
        q, k, v = (restore(name) for name in 'qkv')
        eligible = profile.metal_eligible(q, k, v, None, 0., False)
        if not eligible:
            raise RuntimeError('Captured operation must select Metal attention')
        groups = q.shape[1] // k.shape[1]
        stock = F.scaled_dot_product_attention(q, k.repeat_interleave(groups, dim=1),
                                               v.repeat_interleave(groups, dim=1))
        metal = profile.ar_attention(q, k, v)
        expected = tensors[base + 'output']
        report = {'trace': str(args.trace), 'original_strides_restored': True, 'metal_eligible': bool(eligible)}
        for name, actual in [('stock', stock), ('metal', metal)]:
            actual = actual.transpose(1, 2).cpu().contiguous()
            delta = (actual.float() - expected.float()).abs()
            report[name] = {'exact': torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8)),
                            'unequal': int((actual != expected).sum()), 'elements': actual.numel(),
                            'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['metal']['exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
