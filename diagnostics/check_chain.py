"""Verify every stage of a generated chain against its independent reference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from prepare_validation import workflow


def main():
    p = argparse.ArgumentParser()
    p.add_argument('chain', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, default=Path.home() / 'dev/ai/ComfyUI')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reference = args.reference_root / 'output'
    archive = reference / 'audio/YuE2/_archive'
    cases = [
        ('riff', 'riff45', archive / '2026-09-16_reset/02_takes_45s/swap_B_guitar_00001.flac', None, 0),
        ('grown', 'chain_B_grown', archive / 'B_guitar_GROWN_full.flac', 'riff', 1125),
        ('tail779', 'chain_B_tail779', archive / 'B_guitar_tail_seed779.flac', 'grown', 4500),
        ('master', 'parity_goal_chain_master', archive / '2026-09-16_baseline_reset/29_burn_it_down/burn_01_master.flac', 'tail779', 4950),
    ]
    summary = []
    compare = Path(__file__).with_name('compare_take.py')
    for step, token_name, audio, predecessor, kept in cases:
        take = args.chain / step
        if predecessor:
            actual = np.load(take / 'semantic.npy')
            prior = np.load(args.chain / predecessor / 'semantic.npy')
            if not np.array_equal(actual[:kept], prior[:kept]):
                raise AssertionError(f'{step}: generated predecessor semantic prefix differs')
        report = args.out / (step + '.json')
        subprocess.run([sys.executable, str(compare), str(take),
                        '--tokens', str(reference / 'yue2_takes' / (token_name + '.json')),
                        '--latents', str(reference / 'yue2_takes' / (token_name + '.npy')),
                        '--audio', str(audio), '--out', str(report)], check=True)
        graph = args.out / (step + '_workflow.json')
        graph.write_text(json.dumps(workflow(audio), indent=2) + '\n')
        command = [str(args.reference_root / '.venv/bin/python'),
                   str(Path(__file__).with_name('verify_reference_prefix.py')),
                   '--graph', str(graph), '--actual', str(take / 'prefix.npy'),
                   '--out', str(args.out / (step + '_prefix.json')),
                   '--pack', str(args.reference_root / 'custom_nodes/ComfyUI-FL-YuE2')]
        if step == 'riff':
            command += ['--plan-node', '20']
        subprocess.run(command, check=True)
        summary.append({'step': step, 'exact': True, 'predecessor': predecessor,
                        'carried_tokens': kept, 'report': str(report)})
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
