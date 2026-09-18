"""Compare VAE float waveform output on identical acoustic latents."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
import numpy as np
from trace_ar import load_reference
p=argparse.ArgumentParser();p.add_argument('--stack',choices=['reference','engine'],required=True)
p.add_argument('--latent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--target',type=Path);p.add_argument('--comfy-root',type=Path,default=Path.home()/'dev/ai/ComfyUI')
p.add_argument('--models',type=Path,default=Path.home()/'dev/ai/models/YuE2')
p.add_argument('--odd-output-padding',action='store_true')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);z=np.load(a.latent)
if a.stack=='reference':
 music,modeling,sampling,protocol,vae=load_reference(a.comfy_root,'full')
 runtime=importlib.import_module(modeling.__package__+'.runtime')
 import torch
 with torch.inference_mode():
  result=runtime.decode(vae,torch.tensor(z).T.unsqueeze(0).contiguous(),1024)
  audio=result['waveform'][0].cpu().float().numpy().T
else:
 if a.odd_output_padding:
  from yue2 import modeling_vae
  original = modeling_vae.DecoderBlock.__init__
  def corrected(self, in_channels, out_channels, stride, act_type):
   original(self,in_channels,out_channels,stride,act_type)
   self.layers[1].output_padding = (stride % 2,)
  modeling_vae.DecoderBlock.__init__ = corrected
 from yue2.pipeline import YuE2Pipeline
 pipe=YuE2Pipeline(a.models/'YuE2-3B',a.models/'YuE2-Vae',device='mps',backend='torch',offload_ar=True,progress=False)
 audio=pipe.decode(z)
audio=np.ascontiguousarray(audio);np.save(a.out/'audio.npy',audio)
report=dict(shape=list(audio.shape),sha256=hashlib.sha256(audio.tobytes()).hexdigest())
if a.target:
 expected=np.load(a.target)
 report.update(target_shape=list(expected.shape),exact=False)
 if audio.shape == expected.shape:
  delta=np.abs(audio-expected)
  report.update(exact=np.array_equal(audio.view(np.uint8),expected.view(np.uint8)),unequal=int(np.count_nonzero(audio!=expected)),max_abs=float(delta.max()))
(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
