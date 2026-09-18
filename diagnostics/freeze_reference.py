"""Freeze provenance and small inputs without changing either environment."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    home=Path.home(); project=home/'dev/projects/audiogen'; comfy=home/'dev/ai/ComfyUI'
    roots={'engine_wrapper':home/'dev/projects/audiogen-yue2','engine':home/'dev/projects/YuE',
           'comfy':comfy,'pack':comfy/'custom_nodes/ComfyUI-FL-YuE2',
           'apple_patches':comfy/'custom_nodes/ComfyUI-AppleSilicon-FP8'}
    state={}
    for name,root in roots.items():
        def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True)
        state[name]={'root':str(root),'head':git('rev-parse','HEAD').strip(),'status':git('status','--porcelain'),
                     'source_hashes':{str(f.relative_to(root)):sha(f) for f in root.rglob('*.py')
                                      if not any(x in f.parts for x in ('.venv','.venv-py312.archived','.git','node_modules','out','output')) and not f.is_symlink()}}
        (a.out/(name+'.diff')).write_text(git('diff','HEAD'))
    models=home/'dev/ai/models/YuE2'
    state['model_files']={str(f):sha(f) for f in models.rglob('*') if f.is_file() and f.suffix in ('.safetensors','.json','.tiktoken')}
    files={'prefix.npy':project/'output/step1c_f_rms_norm/baseline01_riff/riff/prefix.npy',
           'greedy45.json':comfy/'output/yue2_takes/greedy45.json',
           'request.json':project/'output/step1c_f_rms_norm/baseline01_riff/riff/request.json',
           'ENV_album.md':project/'ENV_album.md',
           'riff_graph.json':project/'riff_original/full_graph.json'}
    state['inputs']={}
    for name,f in files.items():
        if not f.exists():
            state['inputs'][name]={'missing':str(f)};continue
        shutil.copyfile(f,a.out/name);state['inputs'][name]={'source':str(f),'sha256':sha(f)}
    state['processes']=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
    state['processes']=[line for line in state['processes'] if 'python main.py' in line]
    (a.out/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    print(a.out)

if __name__=='__main__':main()
