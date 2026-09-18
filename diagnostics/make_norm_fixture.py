"""Extract one captured row that distinguishes stock RMSNorm from the reference."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('engine',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('out',type=Path)
a=p.parse_args();r=torch.load(a.reference/'tensors.pt',weights_only=True);e=torch.load(a.engine/'tensors.pt',weights_only=True)
n='0000/model.layers.0.post_attention_layernorm/'
rows=(r[n+'output']!=e[n+'output']).any(-1);index=int(rows[0].nonzero()[0])
with safe_open(a.checkpoint,framework='pt',device='cpu') as f:w=f.get_tensor('model.layers.0.post_attention_layernorm.weight')
a.out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.out/'metal_norm_row.npz',x=r[n+'input'][:,index:index+1].float().numpy(),
                    expected=r[n+'output'][:,index:index+1].float().numpy(),weight=w.float().numpy())
(a.out/'metal_norm_row.json').write_text(json.dumps({'reference':str(a.reference),'engine':str(a.engine),'row':index,'epsilon':1e-6,
 'dtype':'bfloat16','fixture_sha256':hashlib.sha256((a.out/'metal_norm_row.npz').read_bytes()).hexdigest(),
 'description':'One exact captured row from the live-token-validated reference. Float32 storage is an exact widening of BF16.'},indent=2)+'\n')
print(index)
