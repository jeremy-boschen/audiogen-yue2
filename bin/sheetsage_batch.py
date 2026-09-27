#!/usr/bin/env python
"""Transcribe many audio files with SheetSage2, loading the model once.

MUST run under the SheetSage venv (transformers 4.45; this repo's venv is too new for its code):
    ~/dev/ai/sheetsage/.venv/bin/python bin/sheetsage_batch.py OUT_DIR AUDIO...

Each AUDIO gets OUT_DIR/<stem>/ with beat.lab, chord.lab, melody_vocal.lab and the rest.
Existing output directories are skipped.
"""
import sys
import time
import warnings
from pathlib import Path

MODEL = "m-a-p/SheetSage2"
# Pinned: trust_remote_code runs this repo's Python at load time. The validated revision.
REVISION = "eab522a8168e8b8b8c4856bf8609cd86198f01fe"


def main() -> int:
    out, files = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    todo = [f for f in files if not (out / f.stem / "beat.lab").exists()]
    if not todo:
        return 0
    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True,
                                      torch_dtype=torch.float32).to(device).eval()
    for i, path in enumerate(todo, 1):
        t = time.time()
        (out / path.stem).mkdir(parents=True, exist_ok=True)
        model.transcribe(str(path), output_dir=str(out / path.stem), dtype="fp32", melody_only=False)
        print(f"sheetsage [{i}/{len(todo)}] {path.stem} {time.time() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
