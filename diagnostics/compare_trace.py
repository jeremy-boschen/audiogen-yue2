"""Compare two saved traces without loading either model or reference packages."""
import argparse
import json
from pathlib import Path
import torch


def compare(left, right):
    a=json.loads((left/'trace.json').read_text())
    b=json.loads((right/'trace.json').read_text())
    at=torch.load(left/'tensors.pt',map_location='cpu',weights_only=True)
    bt=torch.load(right/'tensors.pt',map_location='cpu',weights_only=True)
    rows=[]
    for name,x in a.items():
        if name not in b:
            rows.append(dict(name=name,status='missing_right')); continue
        y=b[name]
        same=x['shape']==y['shape'] and x['dtype']==y['dtype'] and x['sha256']==y['sha256']
        row=dict(name=name,status='equal' if same else 'different',left=x,right=y)
        if (not same and name in at and name in bt and '/model.norm/' in name
                and at[name].ndim == bt[name].ndim == 3 and at[name].shape[1] == 1
                and at[name].shape[2] == bt[name].shape[2]
                and torch.equal(at[name].contiguous().view(torch.uint8),
                                bt[name][:,-1:].contiguous().view(torch.uint8))):
            row['status'] = 'equal_selected_last_row'
        if not same and name in at and name in bt and at[name].shape==bt[name].shape:
            u,v=at[name],bt[name]
            delta=(u.double()-v.double()).abs()
            row.update(elements=u.numel(),unequal=int((u!=v).sum()),max_abs=float(delta.max()),mean_abs=float(delta.mean()))
        rows.append(row)
    weights_a=json.loads((left/'weights.json').read_text()) if (left/'weights.json').exists() else {}
    weights_b=json.loads((right/'weights.json').read_text()) if (right/'weights.json').exists() else {}
    return dict(left=str(left),right=str(right),weights_equal=weights_a==weights_b if weights_a and weights_b else None,
                weight_differences=[k for k in weights_a.keys()|weights_b.keys() if weights_a.get(k)!=weights_b.get(k)],
                first_difference=next((r for r in rows if r['status']=='different'),None),rows=rows,
                only_right=sorted(b.keys()-a.keys()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=compare(a.left,a.right);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','only_right')},indent=2))
