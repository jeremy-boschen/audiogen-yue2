"""Check actual native artifacts against tokens, latents and decoded reference PCM."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
from yue2.protocol import CODEC_OFFSET

p=argparse.ArgumentParser();p.add_argument('take',type=Path)
p.add_argument('--tokens',type=Path,required=True);p.add_argument('--latents',type=Path,required=True)
p.add_argument('--audio',type=Path,required=True);p.add_argument('--actual-audio',type=Path)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
actual_tokens=np.load(a.take/'semantic.npy').astype(np.int64)+CODEC_OFFSET
expected_tokens=np.asarray(json.loads(a.tokens.read_text())['tokens'],dtype=np.int64)
z=np.load(a.take/'latent.npy');ez=np.load(a.latents)
x,sr=sf.read(a.actual_audio or a.take/'audio.flac',dtype='int32',always_2d=True)
e,esr=sf.read(a.audio,dtype='int32',always_2d=True)
report={'semantic_exact':np.array_equal(actual_tokens,expected_tokens),
        'latents_exact':z.shape==ez.shape and z.dtype==ez.dtype and z.tobytes()==ez.tobytes(),
        'pcm_exact':sr==esr and np.array_equal(x,e),'shape':list(x.shape),'reference_shape':list(e.shape),
        'sample_rate':sr,'reference_sample_rate':esr,'pcm_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
        'reference_pcm_sha256':hashlib.sha256(e.tobytes()).hexdigest(),
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
report['exact']=all(report[k] for k in ('semantic_exact','latents_exact','pcm_exact'))
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['exact'] else 1)
