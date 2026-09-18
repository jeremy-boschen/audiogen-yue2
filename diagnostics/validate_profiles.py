"""Validate engine-profile migration against preserved independent references.

Run only when other MPS inference jobs are idle. Uses existing reference outputs,
never regenerates or overwrites them. Custom stages consume their own predecessors.
"""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--references', type=Path, default=Path('out/parity/validation_v1'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    reference = Path.home() / 'dev/ai/ComfyUI'
    assert not importlib.util.find_spec('comfy') and not importlib.util.find_spec('comfy_kitchen')
    assert not any('/ComfyUI' in path for path in sys.path)
    identity = {'python': sys.executable, 'sys_path': sys.path,
                'engine': json.loads(importlib.metadata.distribution('yue2-infer').read_text('direct_url.json')),
                'no_comfy': True}
    (args.out / 'environment.json').write_text(json.dumps(identity, indent=2) + '\n')

    def run(command, name):
        print(name, flush=True)
        (args.out / (name + '_command.json')).write_text(json.dumps(command, indent=2) + '\n')
        with (args.out / (name + '.log')).open('w') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

    def render(song, out, name):
        run([sys.executable, 'bin/render.py', str(song), '--out', str(out), '--quiet',
             '--profile', 'comfyui-yue2-mps-v1', '--use-pytorch-cross-attention',
             '--disable-smart-memory', '--reserve-vram', '8'], name)

    chain = args.out / 'chain'
    render('songs/burn_it_down_parity', chain, 'chain_render')
    run([sys.executable, 'diagnostics/check_chain.py', str(chain / 'burn_it_down_parity'),
         '--out', str(args.out / 'chain_verified')], 'chain_check')
    reports = []
    for name in ('guitar_seed778', 'guitar_seed780', 'synth_seed777'):
        source = args.references / name
        evidence = json.loads((source / 'comparison_1.json').read_text())
        assert evidence['exact']
        for repeat in (1, 2):
            label = f'{name}_{repeat}'
            destination = args.out / label
            render(Path('songs/parity_validation') / name, destination, label)
            take = destination / name / 'riff'
            report = args.out / (label + '_comparison.json')
            command = [sys.executable, 'diagnostics/compare_take.py', str(take), '--out', str(report)]
            for key in ('tokens', 'latents', 'audio'):
                command += ['--' + key, evidence['arguments'][key]]
            run(command, label + '_check')
            run([str(reference / '.venv/bin/python'), 'diagnostics/verify_reference_prefix.py',
                 '--graph', str(source / 'reference_request.json'), '--actual', str(take / 'prefix.npy'),
                 '--out', str(args.out / (label + '_prefix.json')),
                 '--pack', str(reference / 'custom_nodes/ComfyUI-FL-YuE2')], label + '_prefix_check')
            reports.append({'case': name, 'repeat': repeat, 'exact': True, 'report': str(report)})
            (args.out / 'matrix_summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print('Full chain and six matrix renders match the preserved references exactly.', flush=True)


if __name__ == '__main__':
    main()
