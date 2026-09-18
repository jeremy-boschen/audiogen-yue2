"""Sequential live-reference/custom/repeat validation with retained evidence.

Run only when no other inference job is active. Each reference output gets a
unique name; historical outputs are never overwritten. A submitted job's ID is
persisted before polling and is never automatically resubmitted on timeout.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def api(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def wait_for(api_root, job, destination):
    last_notice = 0
    while True:
        try:
            history = api(api_root + '/history/' + job)
        except (OSError, urllib.error.URLError) as error:
            print(f'{job}: polling interrupted: {error}; retrying same job', flush=True)
            time.sleep(5)
            continue
        if job in history:
            record = history[job]
            status = record.get('status', {})
            if status.get('status_str') == 'error':
                write(destination, record)
                raise RuntimeError(f'Reference job failed: {job}; see {destination}')
            if status.get('completed'):
                write(destination, record)
                return record
        if time.monotonic() - last_notice > 30:
            print(f'Waiting for reference job {job}', flush=True)
            last_notice = time.monotonic()
        time.sleep(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cases', type=Path, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, default=Path.home() / 'dev/ai/ComfyUI')
    p.add_argument('--api', default='http://127.0.0.1:8188')
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    args.out = args.out.resolve()
    if args.repeats < 2:
        p.error('at least two custom runs are required to verify repeatability')
    queue = api(args.api + '/queue')
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise RuntimeError('Reference queue is occupied; wait for the existing work')
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    results = []
    reference_out = args.reference_root / 'output'
    # Time suffix keeps independent runs separate and preserves earlier evidence.
    suffix = str(time.time_ns())
    for case in args.cases:
        case = case.resolve()
        spec = json.loads((case / 'song.json').read_text())
        name = spec['id']
        destination = args.out / name
        destination.mkdir()
        request = copy.deepcopy(json.loads((case / 'reference_request.json').read_text()))
        token_name = f'parity_validation_{name}_{suffix}'
        audio_prefix = f'audio/YuE2/parity_validation/{name}_{suffix}'
        for node in request['prompt'].values():
            if node['class_type'] == 'FL_YuE2_SaveTokens':
                node['inputs']['name'] = token_name
            elif node['class_type'] == 'SaveAudioAdvanced':
                node['inputs']['filename_prefix'] = audio_prefix
        write(destination / 'reference_request.json', request)
        submitted = api(args.api + '/prompt', request)
        write(destination / 'submission.json', submitted)
        print(f'{name}: submitted {submitted["prompt_id"]}', flush=True)
        history = wait_for(args.api, submitted['prompt_id'], destination / 'history.json')
        audio_files = [item for output in history['outputs'].values()
                       for item in output.get('audio', []) if item.get('type') == 'output']
        if len(audio_files) != 1:
            raise RuntimeError(f'Expected one reference audio output, got {audio_files}')
        item = audio_files[0]
        audio = reference_out / item['subfolder'] / item['filename']
        for repeat in range(1, args.repeats + 1):
            custom = destination / f'custom_{repeat}'
            command = [sys.executable, str(root / 'bin/render.py'), str(case), '--out', str(custom),
                       '--quiet', '--use-pytorch-cross-attention', '--disable-smart-memory', '--reserve-vram', '8']
            write(destination / f'command_{repeat}.json', command)
            print(f'{name}: custom run {repeat}/{args.repeats}', flush=True)
            with (destination / f'custom_{repeat}.log').open('w') as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = destination / f'comparison_{repeat}.json'
            subprocess.run([sys.executable, str(root / 'diagnostics/compare_take.py'), str(custom / name / 'riff'),
                            '--tokens', str(reference_out / 'yue2_takes' / (token_name + '.json')),
                            '--latents', str(reference_out / 'yue2_takes' / (token_name + '.npy')),
                            '--audio', str(audio), '--out', str(report)], cwd=root, check=True)
            subprocess.run([str(args.reference_root / '.venv/bin/python'),
                            str(root / 'diagnostics/verify_reference_prefix.py'),
                            '--graph', str(destination / 'reference_request.json'),
                            '--actual', str(custom / name / 'riff/prefix.npy'),
                            '--out', str(destination / f'prefix_{repeat}.json'),
                            '--pack', str(args.reference_root / 'custom_nodes/ComfyUI-FL-YuE2')],
                           cwd=root, check=True)
            results.append({'case': name, 'repeat': repeat, 'comparison': str(report), 'exact': True})
            write(args.out / 'summary.json', results)
    print('All reference and repeat comparisons passed.', flush=True)


if __name__ == '__main__':
    main()
