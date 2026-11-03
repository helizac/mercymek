"""Run upstream motion/depth models remotely, using one persistent private model cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

from mercy import config


def _pending_visibility(error):
    """New private Kaggle resources can briefly return 403/404 to their owner."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        code = getattr(getattr(error, 'response', None), 'status_code', None)
        if code is not None:
            return code in (403, 404)
        error = error.__cause__ or error.__context__
    return False


def _input_fingerprint(job, entries, worker):
    digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode())
    digest.update(worker.encode())
    for relative, path in entries:
        digest.update(relative.encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _wait(api, ref, stage):
    started = time.monotonic()
    deadline = started + 3600
    while time.monotonic() < deadline:
        try:
            state = api.kernels_status(ref)
        except Exception as error:
            if _pending_visibility(error) and time.monotonic() - started < 180:
                time.sleep(10)
                continue
            raise
        print(ref, state, flush=True)
        status = str(state.status).lower()
        if 'complete' in status:
            return
        if any(word in status for word in ['error', 'fail', 'cancel']):
            api.kernels_output(ref, path=str(stage / 'failure'), file_pattern=r'\.log$')
            raise RuntimeError(f'Kaggle job failed: {ref}; see {stage / "failure"}')
        time.sleep(30)
    raise TimeoutError(f'Kaggle still running: {ref}; check status before resubmitting')


def _wait_dataset(api, ref):
    for _ in range(90):
        try:
            if 'ready' in str(api.dataset_status(ref)).lower():
                return
        except Exception as error:
            if not _pending_visibility(error):
                raise
        time.sleep(2)
    raise TimeoutError('Kaggle input dataset not ready')


def run(case: Path, step: str, max_frames=32, force=False):
    output_names = {'recon': ['recon.npz'], 'slowmo': ['slowmo.mp4', 'slowmo.json']}[step]
    if all((case / name).exists() for name in output_names) and not force:
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    owner = api.get_config_value('username')
    stage = config.ROOT / 'data' / 'kaggle-motion' / case.name / step
    stage.mkdir(parents=True, exist_ok=True)
    cache_ref = f'{owner}/mercy-motion-runtime-v1'

    def submit(job, ref, datasets=()):
        folder = stage / job
        folder.mkdir(exist_ok=True)
        (folder / 'worker.py').write_text(f'JOB = {job!r}\n' + (config.ROOT / 'kaggle' / 'motion_worker.py').read_text())
        (folder / 'kernel-metadata.json').write_text(json.dumps({'id': ref, 'title': ref.split('/')[1],
            'code_file': 'worker.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': job != 'setup', 'enable_internet': job == 'setup',
            'dataset_sources': list(datasets), 'kernel_sources': [] if job == 'setup' else [cache_ref]}))
        response = api.kernels_push(str(folder), acc='NvidiaTeslaT4' if job != 'setup' else None)
        if response.error or response.invalid_dataset_sources or response.invalid_kernel_sources:
            raise RuntimeError(str(response))
        print(response, flush=True)

    if not any(k.ref == cache_ref for k in api.kernels_list(mine=True, search='mercy-motion-runtime-v1', page_size=100)):
        submit('setup', cache_ref)
    _wait(api, cache_ref, stage)
    payload_dir = stage / 'payload'
    payload_dir.mkdir(exist_ok=True)
    job = {'case': case.name, 'step': step, 'max_frames': max_frames}
    files = [config.ROOT / 'mercy' / '__init__.py', config.ROOT / 'mercy' / 'config.py',
             config.ROOT / 'mercy' / 'ingest' / '__init__.py', config.ROOT / 'mercy' / 'ingest' / f'{"reconstruct" if step == "recon" else step}.py',
             case / 'meta.json']
    files += sorted((case / 'frames').glob('*.jpg')) if step == 'recon' else [case / 'clip.mp4', case / 'physics.json']
    entries = []
    for file in files:
        relative = Path('data/cases') / case.name / file.relative_to(case) if file.is_relative_to(case) else file.relative_to(config.ROOT)
        entries.append((relative.as_posix(), file))
    fingerprint = _input_fingerprint(job, entries, (config.ROOT / 'kaggle' / 'motion_worker.py').read_text())
    job['input_sha256'] = fingerprint
    with zipfile.ZipFile(payload_dir / 'payload.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('job.json', json.dumps(job))
        for relative, file in entries:
            bundle.write(file, relative)
    # Immutable inputs keep another case's upload from changing a queued job.
    slug = f'mercy-motion-inputs-{fingerprint[:16]}'
    ref = f'{owner}/{slug}'
    (payload_dir / 'dataset-metadata.json').write_text(json.dumps({'id': ref, 'title': 'MERCY motion inputs', 'licenses': [{'name': 'other'}]}))
    exists = any(d.ref == ref for d in api.dataset_list(mine=True, search=slug))
    if not exists:
        response = api.dataset_create_new(str(payload_dir), public=False, quiet=True)
        if getattr(response, 'error', None):
            raise RuntimeError(str(response))
    _wait_dataset(api, ref)
    job_slug = f'mercy-motion-{step}-{fingerprint[:12]}'
    job_ref = f'{owner}/{job_slug}'
    previous = any(k.ref == job_ref for k in api.kernels_list(mine=True, search=job_slug, page_size=100))
    if force or not previous or any(s in str(api.kernels_status(job_ref).status).lower() for s in ['error', 'fail', 'cancel']):
        submit(step, job_ref, [ref])
    _wait(api, job_ref, stage)
    dst = stage / f'{step}-output'
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    api.kernels_output(job_ref, path=str(dst), file_pattern=r'result/(recon\.npz|slowmo\.(mp4|json)|result\.json)$')
    result = json.loads((dst / 'result' / 'result.json').read_text())
    if result['case'] != case.name or result['step'] != step or result.get('input_sha256') != fingerprint:
        raise ValueError('Kaggle motion output belongs to a different job')
    for name in output_names:
        shutil.copy2(dst / 'result' / name, case / name)
    (case / f'{step}.kaggle.json').write_text(json.dumps(dict(result, input_sha256=fingerprint), indent=2))
