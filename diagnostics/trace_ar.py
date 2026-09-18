"""AR parity probe. Reference-only imports are confined to load_reference().

Run with each stack's Python. Never imports reference code for --stack engine.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


def binding(fn):
    try:
        source = inspect.getsourcefile(fn)
    except TypeError:
        source = None
    return dict(module=getattr(fn, '__module__', None), name=getattr(fn, '__qualname__', str(fn)), source=source)


def load_reference(root, boot):
    sys.path.insert(0, str(root))
    sys.argv = ['main.py', '--listen', '0.0.0.0', '--use-pytorch-cross-attention',
                '--disable-smart-memory', '--reserve-vram', '8']
    if boot == 'full':
        import asyncio
        import main as comfy_main
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        import server, nodes, hook_breaker_ac10a0
        prompt_server = server.PromptServer(loop)
        hook_breaker_ac10a0.save_functions()
        loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
        hook_breaker_ac10a0.restore_functions()
        runtime = next(m for n,m in sys.modules.items() if n.endswith('.yue2.runtime') and
                       'ComfyUI-FL-YuE2' in str(getattr(m,'__file__','')))
        base = runtime.__package__
        modeling = importlib.import_module(base+'.model')
        sampling = importlib.import_module(base+'.sampling')
        protocol = importlib.import_module(base+'.protocol')
        music, vae = runtime.load_models(False)
        return music, modeling, sampling, protocol, vae
    import comfy.options
    comfy.options.enable_args_parsing()
    patch = root / 'custom_nodes/ComfyUI-AppleSilicon-FP8/__init__.py'
    spec = importlib.util.spec_from_file_location('parity_apple_patches', patch,
                                                submodule_search_locations=[str(patch.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.path.insert(0, str(root / 'custom_nodes/ComfyUI-FL-YuE2'))
    from yue2 import runtime, model as modeling, sampling, protocol
    music, vae = runtime.load_models(False)
    return music, modeling, sampling, protocol, vae


class Capture:
    def __init__(self, out, phases):
        self.out, self.phases = out, set(phases)
        self.step, self.layer = -1, ''
        self.rows, self.tensors, self.handles = {}, {}, []

    def record(self, name, t):
        if self.step not in self.phases or not torch.is_tensor(t):
            return
        key = f'{self.step:04d}/{name}'
        cpu = t.detach().to('cpu').contiguous()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        self.rows[key] = dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                              sha256=hashlib.sha256(raw).hexdigest())
        if 'layers.' not in name or 'layers.0.' in name:
            self.tensors[key] = cpu

    def install(self, model, modeling, reference):
        def start(m, args, kwargs):
            self.step += 1
            self.record('input_ids', args[0] if args else kwargs['input_ids'])
        self.handles.append(model.register_forward_pre_hook(start, with_kwargs=True))
        for name, mod in model.named_modules():
            if not name or list(mod.children()):
                continue
            def hook(m, args, result, name=name):
                if args:
                    self.record(name + '/input', args[0])
                if isinstance(result, tuple):
                    for i, t in enumerate(result):
                        self.record(name + f'/output{i}', t)
                else:
                    self.record(name + '/output', result)
            self.handles.append(mod.register_forward_hook(hook))
        names = {id(m): n for n, m in model.named_modules()}
        orig = modeling.Attention.project_qkv
        def project(inner, x, cos, sin):
            self.layer = names[id(inner)]
            self.record(self.layer + '/cos', cos)
            self.record(self.layer + '/sin', sin)
            result = orig(inner, x, cos, sin)
            for tag, t in zip('qkv', result):
                self.record(self.layer + '/project_' + tag, t)
            return result
        modeling.Attention.project_qkv = project
        attr = 'attention' if reference else 'sdpa'
        attention = getattr(modeling, attr) if reference else model.profile.ar_attention
        def wrapped(q, k, v, *args, **kwargs):
            for tag, t in zip('qkv', (q, k, v)):
                self.record(self.layer + '/attention_' + tag, t if reference else t.transpose(1, 2))
            result = attention(q, k, v, *args, **kwargs)
            self.record(self.layer + '/attention_output', result if reference else result.transpose(1, 2))
            return result
        if reference:
            setattr(modeling, attr, wrapped)
        else:
            class CapturedProfile:
                def __init__(self, original):
                    self.original = original
                def __getattr__(self, name):
                    return getattr(self.original, name)
                ar_attention = staticmethod(wrapped)
            for module in model.modules():
                if isinstance(module, modeling.Attention):
                    module.profile = CapturedProfile(module.profile)

    def save(self):
        (self.out/'trace.json').write_text(json.dumps(self.rows, indent=2)+'\n')
        torch.save(self.tensors, self.out/'tensors.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-boot', choices=['manual','full'], default='full')
    ap.add_argument('--stack', choices=['reference','engine'], required=True)
    ap.add_argument('--comfy-root', type=Path, default=Path.home()/'dev/ai/ComfyUI')
    ap.add_argument('--models', type=Path, default=Path.home()/'dev/ai/models/YuE2')
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shims', default='norm_c')
    ap.add_argument('--profile', action='store_true', help='use maintained album numerical operations')
    ap.add_argument('--prior', type=Path, help='JSON tokens or NPY model IDs for the carried penalty window; prefix must already include them')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--teacher', action='store_true')
    ap.add_argument('--steps', type=int, default=1125)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original_args = vars(args).copy()
    ids = np.load(args.prefix).ravel().tolist()
    prior = (json.loads(args.prior.read_text())['tokens'] if args.prior.suffix == '.json'
             else np.load(args.prior).ravel().tolist()) if args.prior else []
    target = json.loads(args.target.read_text())['tokens'] if args.target.suffix == '.json' else np.load(args.target).ravel().tolist()
    if args.stack == 'reference':
        music, modeling, sampling, protocol, _vae = load_reference(args.comfy_root, args.reference_boot)
        model = music.prepare(len(ids)+args.steps)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
        from audiogen import shims
        if not args.profile:
            shims.apply(list(filter(None, args.shims.split(','))))
        from yue2 import modeling_yue2 as modeling, sampling, protocol
        from yue2.pipeline import YuE2Pipeline
        pipe = YuE2Pipeline(args.models/'YuE2-3B', args.models/'YuE2-Vae', device='mps',
                           backend='torch', offload_ar=True, progress=False,
                           profile='comfyui-yue2-mps-v1' if args.profile else 'official')
        model = pipe._load_model()
    global torch
    import torch
    import torch.nn.functional as F
    meta = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in original_args.items()},
                torch=torch.__version__, torch_git=torch.version.git_version, python=sys.version,
                platform=platform.platform(), machine=platform.machine(),
                bindings={n:binding(getattr(F,n)) for n in ['rms_norm','scaled_dot_product_attention','linear']},
                model_bindings={
                    'input_norm': binding(model.model.layers[0].input_layernorm.forward),
                    'project_qkv': binding(model.model.layers[0].self_attn.project_qkv),
                    'attention': binding(modeling.attention if args.stack == 'reference' else model.profile.ar_attention),
                    'sampler': binding(sampling.generate_tokens),
                    'distribution': binding(sampling.distribution)},
                packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                environment={k:v for k,v in os.environ.items() if k.startswith(('ASFP8_', 'MTLFLASH', 'PYTORCH_'))})
    (args.out/'runtime.json').write_text(json.dumps(meta, indent=2)+'\n')
    # Compare loaded weights, not just checkpoint paths. Copy before installing hooks.
    weights={}
    for n,t in model.state_dict().items():
        cpu=t.detach().cpu().contiguous()
        weights[n]=dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
    (args.out/'weights.json').write_text(json.dumps(weights,indent=2)+'\n')
    cap = Capture(args.out, [0,1])
    if args.capture:
        cap.install(model, modeling, args.stack=='reference')
        original_distribution = sampling.distribution

        def recorded_distribution(logits, *positional, **keywords):
            cap.record('sampling_logits', logits)
            scores = original_distribution(logits, *positional, **keywords)
            cap.record('scores', scores)
            return scores

        sampling.distribution = recorded_distribution
    samp = protocol.Sampling(temperature=args.temperature,top_p=0.95,top_k=100,
                              repetition_penalty=1.2,penalty_window=50,min_tokens=min(200,args.steps),max_tokens=args.steps)
    with torch.inference_mode():
        if args.teacher:
            c=model.config
            cache=modeling.StaticKVCache(num_layers=c.num_hidden_layers,batch_size=1,
                num_kv_heads=c.num_key_value_heads,max_seq_len=len(ids)+args.steps,
                head_dim=c.head_dim,dtype=next(model.parameters()).dtype,device=torch.device('mps'))
            history=list(prior); tokens=[]
            for step in range(args.steps):
                inp=ids if step==0 else [target[step-1]]
                kw={} if args.stack=='reference' else {'use_cache':True}
                logits=model(torch.tensor([inp],device='mps'),past_key_values=cache,logits_to_keep=1,**kw).logits[:,-1,:]
                scores=sampling.distribution(logits,samp,history,step,'semantic',False)
                cap.record('scores',scores)
                tokens.append(int(scores.argmax()))
                history.append(target[step])
        else:
            kwargs = {'prior': prior} if args.prior else {}
            tokens,timing,truncated=sampling.generate_tokens(model,ids,samp,args.seed,'semantic',**kwargs)
    (args.out/'tokens.json').write_text(json.dumps({'tokens':tokens})+'\n')
    mismatches=[i for i,(a,b) in enumerate(zip(tokens,target)) if a!=b]
    report=dict(tokens=len(tokens),target_tokens=len(target),mismatches=len(mismatches),
                first=mismatches[0] if mismatches else None,exact=tokens==target)
    (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.capture:
        cap.save()
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
