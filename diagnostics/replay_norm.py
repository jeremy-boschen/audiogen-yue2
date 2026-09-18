"""Replay the first differing RMSNorm on identical captured input and weights."""
import argparse,json
from pathlib import Path
import torch
from safetensors import safe_open
from audiogen.vendor import metal_norm

p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();t=torch.load(a.trace/'tensors.pt',map_location='cpu',weights_only=True)
n='model.layers.0.post_attention_layernorm';x=t['0000/'+n+'/input'].to('mps');expected=t['0000/'+n+'/output']
with safe_open(a.checkpoint,framework='pt',device='cpu') as f:w=f.get_tensor(n+'.weight').to('mps')
report={}
with torch.inference_mode():
 for name,actual in [('stock',torch.nn.functional.rms_norm(x,(x.shape[-1],),w,1e-6)),
                     ('metal',metal_norm.fused_rmsnorm_modulate(x.contiguous().view(-1,x.shape[-1]),w,1e-6).view_as(x))]:
  actual=actual.cpu();delta=(actual.float()-expected.float()).abs()
  report[name]={'exact':torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)),
                'unequal':int((actual!=expected).sum()),'max_abs':float(delta.max())}
report['metal_backend']=metal_norm._last_backend
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
