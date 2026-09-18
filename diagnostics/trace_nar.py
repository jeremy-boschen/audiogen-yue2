"""Acoustic-stage parity on fixed semantic tokens; no AR generation."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
import numpy as np
import trace_ar

p=argparse.ArgumentParser()
p.add_argument('--stack',choices=['reference','engine'],required=True)
p.add_argument('--prefix',type=Path,required=True);p.add_argument('--tokens',type=Path,required=True)
p.add_argument('--target',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--comfy-root',type=Path,default=Path.home()/'dev/ai/ComfyUI')
p.add_argument('--models',type=Path,default=Path.home()/'dev/ai/models/YuE2')
p.add_argument('--shims',default='norm_metal,mtl_b');p.add_argument('--capture',action='store_true')
p.add_argument('--seed',type=int,default=777);p.add_argument('--steps',type=int,default=32)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
prefix=np.load(a.prefix).ravel().tolist();tokens=json.loads(a.tokens.read_text())['tokens']
if a.stack=='reference':
 music,modeling,sampling,protocol,_vae=trace_ar.load_reference(a.comfy_root,'full')
 model=music.prepare(len(prefix)+len(tokens))
 nar=importlib.import_module(modeling.__package__+'.nar')
else:
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
 from audiogen import shims
 shims.apply(list(filter(None,a.shims.split(','))))
 from yue2 import modeling_yue2 as modeling,nar,protocol
 from yue2.pipeline import YuE2Pipeline
 pipe=YuE2Pipeline(a.models/'YuE2-3B',a.models/'YuE2-Vae',device='mps',backend='torch',offload_ar=True,progress=False)
 model=pipe._load_model()
import torch
trace_ar.torch=torch
cap=trace_ar.Capture(a.out,[0,1]);cap.step=0
if a.capture:
 cap.install(model,modeling,a.stack=='reference')
 attention=nar.attention
 def attn(q,k,v,*args,**kwargs):
  for tag,t in zip('qkv',(q,k,v)):cap.record(cap.layer+'/attention_'+tag,t[None])
  out=attention(q,k,v,*args,**kwargs)
  cap.record(cap.layer+'/attention_output',out[None]);return out
 nar.attention=attn
 velocity=nar.CachedNAR.velocity
 def velocity_trace(self,*args,**kwargs):
  cap.step+=1
  return velocity(self,*args,**kwargs)
 nar.CachedNAR.velocity=velocity_trace

def progress(i,total):
 if i%8==0:print(f'ODE {i}/{total}',flush=True)
with torch.inference_mode():
 latent=nar.synthesize(model,prefix,[t-protocol.CODEC_OFFSET for t in tokens],a.seed,steps=a.steps,on_progress=progress)
actual=latent.float().cpu().numpy();np.save(a.out/'latent.npy',actual)
expected=np.load(a.target)
delta=np.abs(actual-expected)
result=dict(shape=list(actual.shape),target_shape=list(expected.shape),exact=np.array_equal(actual.view(np.uint8),expected.view(np.uint8)),
 unequal=int(np.count_nonzero(actual!=expected)),max_abs=float(delta.max()),mean_abs=float(delta.mean()),
 sha256=hashlib.sha256(actual.tobytes()).hexdigest())
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(a.out/'arguments.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2)+'\n')
if a.capture:
 cap.save()
print(json.dumps(result),flush=True)
