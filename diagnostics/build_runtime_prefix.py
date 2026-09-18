"""Build the reference Plan-node prefix using only our engine and tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from yue2.protocol import SongRequest, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer

p=argparse.ArgumentParser();p.add_argument('request',type=Path);p.add_argument('tokenizer',type=Path);p.add_argument('out',type=Path)
a=p.parse_args();r=json.loads(a.request.read_text());r['abc']=(r.get('abc') or '').strip() or None
r.pop('id',None)
t=YuE2TextTokenizer(a.tokenizer)
req=SongRequest(**r);ids=t.encode(req.abc) if req.abc is not None else []
prefix=np.asarray(token_prefixes(req,t,ids),dtype=np.int64)
np.save(a.out,prefix)
print(json.dumps({'tokens':len(prefix),'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}))
