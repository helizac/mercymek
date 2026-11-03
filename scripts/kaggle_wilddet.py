"""Run WildDet3D on original RGB frames using persisted, private Kaggle assets/runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config
from scripts.kaggle_gpu import client
from scripts.kaggle_vehicle_pose import http_status

STAGE = ROOT / 'data/kaggle-wilddet'
ASSETS = 'mercy-wilddet-assets-v1'
RUNTIME = 'mercy-wilddet-runtime-v1'


def fetch_log(api, ref, destination):
    # kernels_output enumerates every cached dependency file even with a filename filter.
    # Fetch just the first SDK response for its log; avoid thousands of pagination requests.
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    owner, slug, _ = api.parse_kernel_string(ref)
    request = ApiListKernelSessionOutputRequest()
    request.user_name = owner; request.kernel_slug = slug; request.page_size = 1
    with api.build_kaggle_client() as connection:
        response = connection.kernels.kernels_api_client.list_kernel_session_output(request)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    (destination / (slug + '.log')).write_text(response.log or '', encoding='utf-8')


def push(api, owner, phase, slug, datasets=()):
    folder = STAGE / 'kernels' / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'worker.py').write_text(f'PHASE={phase!r}\n' + (ROOT / 'kaggle/wilddet_worker.py').read_text())
    sources = [] if phase == 'assets' else [f'{owner}/{ASSETS}']
    if phase == 'infer':
        sources.append(f'{owner}/{RUNTIME}')
    (folder / 'kernel-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{slug}', title=slug,
        code_file='worker.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=True, enable_internet=phase == 'assets', dataset_sources=list(datasets), kernel_sources=sources)))
    response = api.kernels_push(str(folder), acc='NvidiaTeslaT4')
    print(response, flush=True)
    if response.error or response.invalid_dataset_sources or response.invalid_kernel_sources:
        raise RuntimeError('Kaggle rejected the WildDet job or its saved dependencies')


def wait(api, ref, output=None):
    start = time.monotonic()
    while time.monotonic() - start < 7200:
        try:
            status = api.kernels_status(ref)
        except Exception as e:
            if http_status(e) in [403, 404] and time.monotonic() - start < 180:
                time.sleep(10)
                continue
            raise
        state = str(status.status).lower()
        print(ref, state, flush=True)
        if 'complete' in state:
            if output:
                api.kernels_output(ref, path=str(output),
                    file_pattern=r'(wilddet-predictions/(results|batch-validation)\.json$|checkpoint-load\.json$)', page_size=100)
            return
        if any(s in state for s in ['error', 'fail', 'cancel']):
            log = STAGE / 'failures' / ref.split('/')[-1]
            log.mkdir(parents=True, exist_ok=True)
            fetch_log(api, ref, log)
            raise RuntimeError(f'WildDet job failed; logs: {log}')
        time.sleep(20)
    raise TimeoutError(f'WildDet job is still pending: {ref}')


def ensure(api, owner, phase, slug, rebuild=False):
    try:
        state = str(api.kernels_status(f'{owner}/{slug}').status).lower()
    except Exception as e:
        if http_status(e) not in [403, 404]:
            raise
        state = 'missing'
    if rebuild or state == 'missing' or any(s in state for s in ['error', 'fail', 'cancel']):
        push(api, owner, phase, slug)
    wait(api, f'{owner}/{slug}')


def prepare(names, smoke=False, hz=2.):
    from mercy.ingest.vehicle_scene import Cameras, continuous_cameras
    from mercy.generation.vehicles import vehicle_observations
    inputs = STAGE / 'frames'
    inputs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in names:
        case = config.case_dir(name)
        meta = json.loads((case / 'meta.json').read_text())
        overlays = json.loads((case / 'overlays.json').read_text())
        first = json.loads((case / 'moments.json').read_text())['moments'][0]
        with np.load(case / 'recon.npz') as rec:
            times = rec['times']; K = rec['intrinsic'].copy(); rh, rw = rec['depth'].shape[1:]
        K *= np.array([[meta['width'] / rw], [meta['height'] / rh], [1.]])
        with np.load(case / 'environment/cameras.npz') as cache:
            matrices = cache['matrices']
        cameras = Cameras(times, continuous_cameras(times, matrices, first['replay_start'], first['replay_end']), K)
        available = sorted({f for rows in vehicle_observations(overlays).values() for f, _ in rows
            if first['replay_start'] <= f / meta['src_fps'] <= first['replay_end']})
        target = ([.95, 4.02, 4.56] if name == 'cutin_rearend' else [5.5, 8., 9.3]) if smoke else np.arange(first['replay_start'], first['replay_end'] + .001, 1 / hz)
        frames = sorted({min(available, key=lambda f: abs(f / meta['src_fps'] - t)) for t in target})
        cap = cv2.VideoCapture(str(case / 'clip.mp4'))
        source_hash = hashlib.sha256((case / 'overlays.json').read_bytes()).hexdigest()
        try:
            for frame in frames:
                objects = [dict(track=k, box=v['box']) for k, v in overlays[str(frame)].items()
                    if v.get('concept', '').lower() in {'car', 'truck', 'bus', 'van', 'motorcycle', 'motorbike', 'pickup', 'vehicle'}
                    and v['box'][2] - v['box'][0] >= 10 and v['box'][3] - v['box'][1] >= 10]
                if not objects:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Cannot read original source {name}:{frame}')
                path = inputs / f'{name}-{frame:05d}.png'
                cv2.imwrite(str(path), image)
                C, R, lens = cameras.at(frame / meta['src_fps'])
                jobs.append(dict(case=name, frame=frame, t=frame / meta['src_fps'], image=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), overlays_sha256=source_hash,
                    K=lens.tolist(), C=C.tolist(), R=R.tolist(), objects=objects))
        finally:
            cap.release()
    return inputs, jobs


def run(api, owner, names, smoke=False, hz=2., publish=True):
    inputs, jobs = prepare(names, smoke, hz)
    fingerprint = hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()[:12]
    upload = STAGE / 'upload' / fingerprint
    upload.mkdir(parents=True, exist_ok=True)
    ref = f'{owner}/mercy-wilddet-inputs-{fingerprint}'
    with zipfile.ZipFile(upload / 'wilddet-inputs.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('wilddet-jobs.json', json.dumps(jobs))
        for j in jobs:
            bundle.write(inputs / j['image'], j['image'])
    (upload / 'dataset-metadata.json').write_text(json.dumps(dict(id=ref, title='MERCY original WildDet RGB inputs', licenses=[dict(name='other')])))
    if not any(d.ref == ref for d in api.dataset_list(mine=True, search=ref.split('/')[-1])):
        response = api.dataset_create_new(str(upload), public=False, quiet=True)
        if getattr(response, 'error', None):
            raise RuntimeError(str(response))
    for _ in range(90):
        try:
            if 'ready' in str(api.dataset_status(ref)).lower():
                break
        except Exception as e:
            if http_status(e) not in [403, 404]:
                raise
        time.sleep(2)
    else:
        raise TimeoutError('Private WildDet input dataset is not ready')
    execution = hashlib.sha256(fingerprint.encode() + (ROOT / 'kaggle/wilddet_worker.py').read_bytes()).hexdigest()[:12]
    slug = f'mercy-wilddet-infer-{execution}'
    output = STAGE / 'output' / execution
    output.mkdir(parents=True, exist_ok=True)
    latest = dict(kernel=f'{owner}/{slug}', output=output.relative_to(ROOT).as_posix(), frames=len(jobs))
    (STAGE / 'latest.json').write_text(json.dumps(latest, indent=2))
    if not (output / 'wilddet-predictions/results.json').exists():
        try:
            state = str(api.kernels_status(f'{owner}/{slug}').status).lower()
        except Exception as e:
            if http_status(e) not in [403, 404]:
                raise
            state = 'missing'
        if state == 'missing' or any(s in state for s in ['error', 'fail', 'cancel']):
            push(api, owner, 'infer', slug, [ref])
        wait(api, f'{owner}/{slug}', output)
    print('WildDet output:', output, flush=True)
    if publish:
        from mercy.ingest.wilddet import import_results
        from mercy.ingest.vehicle_scene import publish as publish_scene
        from mercy.analysis.seal import seal
        results = json.loads((output / 'wilddet-predictions/results.json').read_text())
        for name in names:
            case = config.case_dir(name)
            report = import_results(case, results)
            publish_scene(case)
            seal(case)
            print(name, len(report['accepted']), '/', report['predictions'], 'qualified WildDet anchors published', flush=True)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['assets', 'runtime', 'run', 'status'])
    p.add_argument('--case', nargs='+')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--hz', type=float, default=2.)
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--no-publish', action='store_true', help='diagnostic inference only; do not update scene manifests')
    args = p.parse_args()
    api = client(); owner = api.get_config_value('username')
    if args.action == 'assets':
        ensure(api, owner, 'assets', ASSETS, args.rebuild)
    elif args.action == 'runtime':
        ensure(api, owner, 'assets', ASSETS)
        ensure(api, owner, 'runtime', RUNTIME, args.rebuild)
    elif args.action == 'run':
        if not args.case:
            p.error('run requires --case')
        ensure(api, owner, 'assets', ASSETS)
        ensure(api, owner, 'runtime', RUNTIME)
        run(api, owner, args.case, args.smoke, args.hz, not args.no_publish)
    else:
        for slug in [ASSETS, RUNTIME]:
            try:
                print(slug, api.kernels_status(f'{owner}/{slug}'), flush=True)
            except Exception as e:
                print(slug, type(e).__name__, http_status(e), flush=True)
