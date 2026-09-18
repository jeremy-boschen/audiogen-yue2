"""Record application and engine numerical provenance; no runtime patching."""
from pathlib import Path
import hashlib
import platform
from importlib.metadata import version


def runtime_identity(profile):
    import av
    import torch
    return {
        'profile': profile.name,
        'profile_identity': profile.identity(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'torch_git': torch.version.git_version,
        'packages': {name: version(name) for name in ('torch', 'mtlflashattn', 'av', 'yue2-infer')},
        'libav': {name: list(value) for name, value in av.library_versions.items()},
        'source_sha256': {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.rglob('*.py'))},
    }
