#!/usr/bin/env python3
"""How much does writing several takes at once cost, per code, on this Mac?

    bin/bench_batch.py [--batches 1,2,4,8] [--contexts 1500,5000] [--steps 150] [--out FILE]

Loads the model as the studio does (MPS, comfyui-yue2-mps-v1 profile, bf16), fills a
cache with a random prefix of each length, then times the writing step: sample a code
per take, read it back to the CPU as the real loop does, run the next forward. Random
codes cost the same as real ones; only speed is measured, not what is written.

Batch 2 also stands in for guidance, whose two branches the Mac runs one after the other.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--contexts", default="1500,5000")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--models", default=str(HERE / "models"))
    parser.add_argument("--out")
    args = parser.parse_args()
    from yue2.modeling_yue2 import StaticKVCache
    from yue2.pipeline import YuE2Pipeline

    models = pathlib.Path(args.models)
    pipe = YuE2Pipeline(models / "YuE2-3B", models / "YuE2-Vae", device="mps", profile="comfyui-yue2-mps-v1", progress=False)
    model = pipe._load_model()
    config = model.config
    device = next(model.parameters()).device

    def run(batch: int, context: int, sync: bool) -> float:
        torch.manual_seed(0)
        ids = torch.randint(1000, 150000, (batch, context), device=device)
        cache = StaticKVCache(num_layers=config.num_hidden_layers, batch_size=batch, num_kv_heads=config.num_key_value_heads,
                              max_seq_len=context + args.steps + 16, head_dim=config.head_dim, dtype=torch.bfloat16, device=device)
        with torch.inference_mode():
            logits = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :]
            torch.mps.synchronize()
            start = 0.0
            for step in range(args.steps + 10):
                if step == 10:  # the first steps warm the kernels up
                    torch.mps.synchronize()
                    start = time.perf_counter()
                probabilities = logits.float().softmax(-1)
                if sync:  # as the studio does: sampled on the CPU, one code per take read back
                    codes = torch.multinomial(probabilities.cpu(), 1).to(device)
                else:
                    codes = torch.multinomial(probabilities, 1)
                logits = model(codes, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :]
            torch.mps.synchronize()
        del cache
        torch.mps.empty_cache()
        return (time.perf_counter() - start) / args.steps

    rows = []
    for context in [int(c) for c in args.contexts.split(",")]:
        for batch in [int(b) for b in args.batches.split(",")]:
            for sync in ([True, False] if batch == 1 else [True]):
                seconds = run(batch, context, sync)
                row = {"context": context, "batch": batch, "cpu_sampling": sync, "ms_per_step": round(seconds * 1000, 2),
                       "codes_per_s_per_take": round(1 / seconds, 1), "codes_per_s_total": round(batch / seconds, 1)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"device": torch.backends.mps.is_available() and "mps", "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
