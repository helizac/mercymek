"""Kaggle CUDA worker for upstream VGGT and RAFT. JOB is prepended at submission.

The private setup output is reused by each new case. No project .env is uploaded.
"""
from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

INPUT, WORK = Path('/kaggle/input'), Path('/kaggle/working')

if JOB == 'setup':
    cache = WORK / 'motion-runtime'
    cache.mkdir(exist_ok=True)
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/vggt.git', str(cache / 'vggt')], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(cache / 'site'),
                    'einops==0.8.1', 'python-dotenv==1.1.1'], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download('facebook/VGGT-1B', local_dir=cache / 'vggt-weights', allow_patterns=['*.json', '*.safetensors'])
    os.environ['TORCH_HOME'] = str(cache / 'torch')
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft_large(weights=Raft_Large_Weights.DEFAULT)
    (cache / 'manifest.json').write_text(json.dumps({'vggt': 'facebook/VGGT-1B', 'raft': 'torchvision RAFT-large'}))
else:
    cache = next(INPUT.rglob('motion-runtime/manifest.json')).parent
    sys.path[:0] = [str(cache / 'site'), str(cache / 'vggt')]
    os.environ['TORCH_HOME'] = str(cache / 'torch')
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if not torch.cuda.is_available():
        raise RuntimeError('A Kaggle GPU is required')
    payload = next(INPUT.rglob('payload.zip'), None)
    project = WORK / 'project'
    if payload:
        with zipfile.ZipFile(payload) as bundle:
            for member in bundle.infolist():
                if not (project / member.filename).resolve().is_relative_to(project.resolve()):
                    raise ValueError('Unsafe archive')
            bundle.extractall(project)
    else:
        shutil.copytree(next(INPUT.rglob('job.json')).parent, project)
    sys.path.insert(0, str(project))
    job = json.loads((project / 'job.json').read_text())
    case = project / 'data' / 'cases' / job['case']
    dst = WORK / 'result'
    dst.mkdir()
    if JOB == 'recon':
        import numpy as np
        from mercy.ingest.reconstruct import run_vggt
        # Safetensors performs many parameter reads. Stage the immutable cache
        # sequentially on local storage instead of reading every tensor through
        # Kaggle's attached-output filesystem during model loading.
        staged_at = time.monotonic()
        weights = Path('/tmp/mercy-vggt-weights')
        shutil.copytree(cache / 'vggt-weights', weights, ignore=shutil.ignore_patterns('.cache'))
        print(f'[vggt] cached weights staged locally in {time.monotonic()-staged_at:.1f}s', flush=True)
        result = run_vggt(case, max_frames=job['max_frames'], model_id=str(weights))
        np.savez_compressed(dst / 'recon.npz', **result)
    elif JOB == 'slowmo':
        from mercy.ingest.slowmo import build
        if build(case, force=True) is None:
            raise RuntimeError('No slow motion produced')
        for name in ['slowmo.mp4', 'slowmo.json']:
            shutil.copy2(case / name, dst / name)
    else:
        raise ValueError(JOB)
    (dst / 'result.json').write_text(json.dumps(dict(job, gpu=torch.cuda.get_device_name(0))))
