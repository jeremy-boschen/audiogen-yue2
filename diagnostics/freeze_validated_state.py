"""Bind a validated take to current sources, locked packages and reference state."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--take', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frozen', type=Path, default=Path('out/parity/frozen/manifest.json'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((args.take / 'config.json').read_text())
    recorded = config['audiogen']['numerics_identity']['source_sha256']
    changed = [name for name, expected in recorded.items() if sha(root / 'src/audiogen' / name) != expected]
    if changed:
        raise RuntimeError(f'Production source differs from validated take: {changed}')
    packages = {d.metadata['Name']: d.version for d in metadata.distributions()}
    for line in (root / 'env/requirements.in').read_text().splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            name, expected = line.strip().split('==')
            if metadata.version(name) != expected:
                raise RuntimeError(f'Locked version mismatch: {name}')
    comfy = [name for name in packages if 'comfy' in name.lower()]
    if comfy or importlib.util.find_spec('comfy') or importlib.util.find_spec('comfy_kitchen'):
        raise RuntimeError(f'ComfyUI is importable or installed in the custom environment: {comfy}')
    if any('/ComfyUI' in path for path in sys.path):
        raise RuntimeError('Reference tree appears on the custom import path')
    engine = json.loads(metadata.distribution('yue2-infer').read_text('direct_url.json'))
    pin = next(line.split('=', 1)[1] for line in (root / 'env/pins.env').read_text().splitlines()
               if line.startswith('YUE_COMMIT='))
    if engine.get('vcs_info', {}).get('commit_id') != pin:
        raise RuntimeError('Installed engine does not match the pinned git commit')
    import yue2.pipeline
    from yue2.storage import identity
    engine_root = Path(yue2.pipeline.__file__).parent
    engine_sources = {str(path.relative_to(engine_root)): sha(path)
                      for path in sorted(engine_root.rglob('*')) if path.suffix in ('.py', '.json')}
    if identity(engine_sources) != config['runtime_sha256']:
        raise RuntimeError('Installed engine source differs from the validated take')
    frozen = json.loads(args.frozen.read_text())
    model_changes = [path for path, expected in frozen['model_files'].items() if sha(Path(path)) != expected]
    if model_changes:
        raise RuntimeError(f'Frozen model/config/tokenizer files changed: {model_changes}')
    reference_changes = {}
    reference_git = {}
    for name in ('comfy', 'pack', 'apple_patches'):
        state = frozen[name]
        head = subprocess.check_output(['git', '-C', state['root'], 'rev-parse', 'HEAD'], text=True).strip()
        if head != state['head']:
            raise RuntimeError(f'Reference revision changed: {name}')
        reference_git[name] = {'head': head, 'status': subprocess.check_output(
            ['git', '-C', state['root'], 'status', '--porcelain'], text=True)}
        reference_changes[name] = [path for path, expected in state['source_hashes'].items()
                                   if not (Path(state['root']) / path).is_file()
                                   or sha(Path(state['root']) / path) != expected]
    if any(reference_changes.values()):
        raise RuntimeError(f'Reference source changed: {reference_changes}')
    files = [path for folder in ('src/audiogen', 'bin', 'env', 'songs', 'tests', 'diagnostics')
             for path in (root / folder).rglob('*')
             if path.is_file() and '__pycache__' not in path.parts
             and path.suffix in ('.py', '.json', '.abc', '.txt', '.npz', '.env', '.in', '.md', '.sh', '.toml')]
    files += [root / name for name in ('pyproject.toml', 'setup.sh')]
    result = {
        'validated_take': str(args.take), 'validated_config_sha256': sha(args.take / 'config.json'),
        'production_sources_unchanged': True, 'reference_source_changes': reference_changes,
        'model_files_verified': frozen['model_files'],
        'reference_git': reference_git,
        'engine': engine, 'engine_source_sha256': engine_sources, 'packages': packages, 'executable': sys.executable, 'sys_path': sys.path,
        'platform': platform.platform(), 'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'files': {str(path.relative_to(root)): sha(path) for path in sorted(files)},
    }
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Validated source/environment state frozen to {args.out}')


if __name__ == '__main__':
    main()
