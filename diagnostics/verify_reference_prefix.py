"""Reference-only prefix reconstruction from an independent embedded workflow.

Run in the reference interpreter; no GPU model is loaded. Production code never
imports these reference modules. This verifies request/prefix equality separately
from downstream token or PCM equality.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--plan-node')
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pack', type=Path, default=Path.home() / 'dev/ai/ComfyUI/custom_nodes/ComfyUI-FL-YuE2')
    p.add_argument('--tokenizer', type=Path, default=Path.home() / 'dev/ai/models/YuE2/YuE2-3B/qwen.tiktoken')
    args = p.parse_args()
    graph = json.loads(args.graph.read_text())
    graph = graph.get('prompt', graph)
    plans = [(key, node['inputs']) for key, node in graph.items()
             if node['class_type'] == 'FL_YuE2_Plan' and (args.plan_node is None or args.plan_node == key)]
    if len(plans) != 1:
        p.error('select exactly one Plan node with --plan-node')
    key, inputs = plans[0]
    if not inputs['score_abc'].strip():
        p.error('this probe requires the supplied-score fixtures')
    # Invoke the actual reference normalization boundary, rather than duplicating
    # its behavior in the verifier (which previously concealed a lyric trim bug).
    sys.path[:0] = [str(args.pack.resolve()), str(args.pack.resolve().parents[1])]
    sys.argv = ['reference_prefix']
    runtime = importlib.import_module('yue2.runtime')
    if Path(runtime.__file__).resolve() != (args.pack / 'yue2/runtime.py').resolve():
        raise RuntimeError('Imported the wrong reference runtime')
    music = SimpleNamespace(tokenizer=runtime.YuE2TextTokenizer(args.tokenizer))
    plan = runtime.make_plan(music, inputs['style'], inputs['lyrics'], inputs['seed'],
                             inputs['planning'], inputs['score_abc'], 12000)
    expected = np.asarray(plan.prefix, dtype=np.int64)
    actual = np.load(args.actual)
    result = {'exact': np.array_equal(expected, actual), 'expected_tokens': expected.size,
              'actual_tokens': actual.size, 'plan_node': key, 'entrypoint': 'yue2.runtime.make_plan',
              'expected_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
              'actual_sha256': hashlib.sha256(actual.astype(np.int64).tobytes()).hexdigest(),
              'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (args.graph, args.pack / 'yue2/runtime.py', args.pack / 'yue2/protocol.py',
                                       args.pack / 'yue2/tokenizer.py', args.tokenizer)}}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
