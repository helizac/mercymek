"""WildDet3D on Kaggle. Immutable assets -> reusable runtime -> offline RGB inference."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

WORK = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
REVISIONS = {
    'wilddet': ('allenai/WildDet3D', '1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a'),
    'sam3': ('weikaih04/sam3_for_wilddet3d', '159490bb13108749dd5ceff68f42cd1792e60641'),
    'lingbot': ('Robbyant/lingbot-depth', 'ff29de5bf433d1b1519cd851da00bc652668964a'),
    'utils3d': ('EasternJournalist/utils3d', 'eeafc2786ec603b2e4f491538c8f2583637905f5'),
    'cuda_ops': ('SysCV/vis4d_cuda_ops', 'd2478d80045b62f85be19235981a63ea9bc8db66'),
}
WEIGHT_REV = '161edb26239f58d8973045517d9c8b179c515f73'
LINGBOT_REV = 'd1b06f7538d4b56387412799c528542c795df169'
CHECKPOINT = 'wilddet3d_alldata_all_prompt_v1.0.pt'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    print('Downloading', path.name, flush=True)
    urllib.request.urlretrieve(url, path)


def source(name, destination):
    repo, revision = REVISIONS[name]
    archive = WORK / (name + '.tar.gz')
    download(f'https://codeload.github.com/{repo}/tar.gz/{revision}', archive)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts or member.issym() or member.islnk():
                continue
            member.name = str(Path(*parts))
            if not (destination / member.name).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe source archive')
            bundle.extract(member, destination, filter='data')
    archive.unlink()


def assets():
    cache = WORK / 'wilddet-assets'
    cache.mkdir()
    source('wilddet', cache / 'source')
    source('sam3', cache / 'source/third_party/sam3')
    source('lingbot', cache / 'source/third_party/lingbot_depth')
    source('utils3d', cache / 'utils3d')
    source('cuda_ops', cache / 'cuda_ops')
    site = cache / 'site'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(site),
        'vis4d==1.0.0', 'timm==1.0.22', 'ftfy==6.1.1', 'iopath==0.1.10',
        'terminaltables==3.1.10', 'ml_collections==1.1.0', 'pyquaternion==0.9.9',
        'setuptools==79.0.1', 'portalocker==3.2.0', 'plyfile==1.1.3', 'imagesize==1.4.1'], check=True)
    download(f'https://huggingface.co/allenai/WildDet3D/resolve/{WEIGHT_REV}/{CHECKPOINT}', cache / CHECKPOINT)
    lingbot = cache / 'lingbot-full.pt'
    download(f'https://huggingface.co/robbyant/lingbot-depth-postrain-dc-vitl14/resolve/{LINGBOT_REV}/model.pt', lingbot)
    import torch
    torch.set_num_threads(2)
    config = torch.load(lingbot, map_location='cpu', weights_only=True)['model_config']
    torch.save({'model_config': config}, cache / 'lingbot-config.pt')
    lingbot.unlink()  # Full WildDet checkpoint already includes these weights; only constructor config is needed.
    manifest = {'sources': REVISIONS, 'weights_revision': WEIGHT_REV, 'lingbot_config_revision': LINGBOT_REV,
        'checkpoint_sha256': digest(cache / CHECKPOINT), 'python': sys.version, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'checkpoint': CHECKPOINT, 'external_sensors': False}
    (cache / 'assets.json').write_text(json.dumps(manifest, indent=2))
    print('Immutable WildDet assets saved', manifest, flush=True)


def paths(assets, runtime):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['MERCY_LINGBOT_CONFIG'] = str(assets / 'lingbot-config.pt')
    os.environ['MERCY_LOAD_REPORT'] = str(WORK / 'checkpoint-load.json')
    sys.path[:0] = [str(runtime / 'source'), str(runtime / 'site'), str(assets / 'site'), str(assets / 'utils3d')]


def build(assets):
    runtime = WORK / 'wilddet-runtime'
    runtime.mkdir(exist_ok=True)
    shutil.copytree(assets / 'source', runtime / 'source', dirs_exist_ok=True)
    # Source assets (videos and presentation images) are irrelevant to inference.
    shutil.rmtree(runtime / 'source/assets', ignore_errors=True)
    p = runtime / 'source/wilddet3d/inference.py'
    code = p.read_text()
    old = 'pretrained_model="robbyant/lingbot-depth-postrain-dc-vitl14"'
    assert old in code
    code = code.replace(old, 'pretrained_model=__import__("os").environ["MERCY_LINGBOT_CONFIG"]')
    # Upstream creates N prompt outputs for one shared image, but only unwraps B image outputs.
    assert code.count('for i in range(B):') == 1
    code = code.replace('for i in range(B):', 'for i in range(len(input_boxes) if input_boxes is not None else B):')
    old = 'wilddet3d.load_state_dict(new_state_dict, strict=False)'
    assert old in code
    code = code.replace(old, '''load_result = wilddet3d.load_state_dict(new_state_dict, strict=False)
    import json, os
    load_report = dict(missing=list(load_result.missing_keys), unexpected=list(load_result.unexpected_keys))
    __import__('pathlib').Path(os.environ['MERCY_LOAD_REPORT']).write_text(json.dumps(load_report, indent=2))
    if load_result.missing_keys:
        raise RuntimeError(f'Checkpoint is missing model parameters: {load_result.missing_keys}')''')
    p.write_text(code)
    # Box-prompt inference uses torchvision's 2D NMS, not the training/evaluation 3D-IoU operator.
    # Keep the real operator as a lazy import; no substitute values or CPU model fallback are introduced.
    for relative in ['wilddet3d/ops/box3d.py', 'wilddet3d/ops/iou_box3d.py']:
        op_file = runtime / 'source' / relative
        op_code = op_file.read_text()
        assert 'from vis4d_cuda_ops import iou_box3d' in op_code
        op_code = op_code.replace('from vis4d_cuda_ops import iou_box3d', '''def iou_box3d(*args, **kwargs):
    from vis4d_cuda_ops import iou_box3d as actual_iou_box3d
    return actual_iou_box3d(*args, **kwargs)
''')
        op_file.write_text(op_code)
    paths(assets, runtime)
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle CUDA GPU required')
    from wilddet3d import build_model
    model = build_model(str(assets / CHECKPOINT), skip_pretrained=True, nms=False, device='cuda')
    manifest = dict(json.loads((assets / 'assets.json').read_text()), gpu=torch.cuda.get_device_name(0),
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        patches=['offline LingBot constructor config', 'unwrap each shared-image geometric prompt',
                 'reject missing checkpoint parameters', 'lazy import unused training/evaluation 3D IoU operator'],
        custom_3d_iou_available=False,
        load_report=json.loads((WORK / 'checkpoint-load.json').read_text()))
    (runtime / 'runtime.json').write_text(json.dumps(manifest, indent=2))
    print('Runtime imported, full checkpoint loaded on GPU; unused training-only 3D IoU is lazy', flush=True)
    del model


def infer(assets, runtime):
    paths(assets, runtime)
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from wilddet3d import build_model, preprocess
    if not torch.cuda.is_available():
        raise RuntimeError('GPU inference required; refusing CPU fallback')
    manifest = json.loads((runtime / 'runtime.json').read_text())
    if manifest['torch'] != torch.__version__ or manifest['python'].split()[0] != sys.version.split()[0]:
        raise RuntimeError('Saved runtime is incompatible with the current Kaggle image')
    src = next(INPUT.rglob('wilddet-jobs.json'), None)
    if src is None:
        archive = next(INPUT.rglob('wilddet-inputs.zip'))
        src_dir = WORK / 'inputs'
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not (src_dir / member.filename).resolve().is_relative_to(src_dir.resolve()):
                    raise ValueError('Unsafe payload')
            bundle.extractall(src_dir)
        src = src_dir / 'wilddet-jobs.json'
    jobs = json.loads(src.read_text())
    model = build_model(str(assets / CHECKPOINT), skip_pretrained=True, nms=False, device='cuda')
    # The network shares image features across prompts, but its default postprocessor merges them by
    # image ID. Split only the postprocessing metadata so each source box keeps its own proposal group.
    # Tensor.expand is a view here; the image backbone/depth model still ran exactly once.
    import copy
    import types
    original_test = model.wilddet3d._forward_test
    def per_prompt_test(self, *args, **kwargs):
        batch = kwargs['batch']
        n = len(batch.img_ids)
        if batch.images.shape[0] == 1 and n > 1:
            batch = copy.copy(batch)
            batch.images = batch.images.expand(n, -1, -1, -1)
            batch.intrinsics = batch.intrinsics.expand(n, -1, -1)
            batch.img_ids = torch.arange(n, device=batch.images.device)
            kwargs['batch'] = batch
            geometry = kwargs.get('geom_out')
            if geometry is not None:
                geometry = dict(geometry)
                for key in ['depth_map', 'confidence_map', 'K_pred']:
                    value = geometry.get(key)
                    if torch.is_tensor(value) and value.shape[0] == 1:
                        geometry[key] = value.expand(n, *value.shape[1:])
                kwargs['geom_out'] = geometry
        return original_test(*args, **kwargs)
    model.wilddet3d._forward_test = types.MethodType(per_prompt_test, model.wilddet3d)
    manifest = dict(manifest, inference_patches=['preserve each shared-image prompt as its own postprocessing group'])
    out = WORK / 'wilddet-predictions'
    out.mkdir()
    results = []
    batch_validated = False
    for job in jobs:
        path = src.parent / job['image']
        if digest(path) != job['sha256']:
            raise ValueError('Source image hash mismatch')
        image = cv2.imread(str(path))
        data = preprocess(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32), np.array(job['K'], np.float32))
        predictions = []
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        # Conservative prompt chunks keep the 1.2B model inside T4 memory without reducing input resolution.
        for offset in range(0, len(job['objects']), 3):
            objects = job['objects'][offset:offset + 3]
            n = len(objects)
            with torch.inference_mode():
                result = model(images=data['images'].cuda(), intrinsics=data['intrinsics'].cuda()[None],
                    input_hw=[data['input_hw']] * n, original_hw=[data['original_hw']] * n,
                    padding=[data['padding']] * n, input_boxes=[o['box'] for o in objects], prompt_text='geometric')
            boxes, boxes3d, scores, scores2d, scores3d, class_ids, depth = result[:7]
            if len(boxes) != n:
                raise RuntimeError('Geometric prompts lost their one-to-one source association')
            if not batch_validated and n > 1:
                with torch.inference_mode():
                    single = model(images=data['images'].cuda(), intrinsics=data['intrinsics'].cuda()[None],
                        input_hw=[data['input_hw']], original_hw=[data['original_hw']], padding=[data['padding']],
                        input_boxes=[objects[0]['box']], prompt_text='geometric')
                paired, independent = boxes3d[0][0].float(), single[1][0][0].float()
                if not torch.allclose(boxes[0][0], single[0][0][0], rtol=.001, atol=.5):
                    raise RuntimeError('Batched 2D prompt output disagrees with independent inference')
                if not torch.allclose(paired[:6], independent[:6], rtol=.003, atol=.005):
                    raise RuntimeError('Batched 3D prompt output disagrees with independent inference')
                if abs(float(torch.dot(paired[6:], independent[6:]))) < .999:
                    raise RuntimeError('Batched quaternion disagrees with independent inference')
                validation = dict(prompts=n, independent_comparison=True,
                    max_box3d_parameter_error=float((paired[:6] - independent[:6]).abs().max()),
                    quaternion_dot_abs=abs(float(torch.dot(paired[6:], independent[6:]))))
                (out / 'batch-validation.json').write_text(json.dumps(validation, indent=2))
                print('BATCH PARITY VERIFIED', validation, flush=True)
                batch_validated = True
                del single
            for i, obj in enumerate(objects):
                predictions.append(dict(track=obj['track'], source_box=obj['box'],
                    box=boxes[i][0].float().cpu().tolist(), box3d=boxes3d[i][0].float().cpu().tolist(),
                    score=float(scores[i][0]), score2d=float(scores2d[i][0]), score3d=float(scores3d[i][0])))
            del result, depth
        record = dict(job, predictions=predictions, seconds=time.monotonic() - start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(), runtime=manifest)
        name = f'{job["case"]}-{job["frame"]:05d}'
        (out / (name + '.json')).write_text(json.dumps(record, indent=2))
        results.append(record)
        print('DONE', name, len(predictions), 'objects', round(record['seconds'], 1), 'seconds', flush=True)
    (out / 'results.json').write_text(json.dumps(results, indent=2))


if PHASE == 'assets':
    assets()
else:
    asset_dir = next(INPUT.rglob('wilddet-assets/assets.json')).parent
    if PHASE == 'runtime':
        build(asset_dir)
    elif PHASE == 'infer':
        runtime_dir = next(INPUT.rglob('wilddet-runtime/runtime.json')).parent
        infer(asset_dir, runtime_dir)
    else:
        raise ValueError(PHASE)
