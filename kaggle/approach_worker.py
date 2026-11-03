"""Private Kaggle guide/inference jobs for a backwards road extension.

PHASE is prepended by scripts/extend_approach.py. Reuses the saved SHARP/gsplat runtime offline.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import zipfile

import numpy as np
import torch
from PIL import Image

ROOT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
cache = next(ROOT.rglob('mercy-runtime/manifest.json')).parent
runtime = json.loads((cache / 'manifest.json').read_text())
if runtime['torch'] != torch.__version__ or runtime['python'].split()[0] != sys.version.split()[0]:
    raise RuntimeError('The saved GPU runtime is incompatible with this Kaggle image')
if not torch.cuda.is_available():
    raise RuntimeError('A Kaggle CUDA GPU is required')
sys.path[:0] = [str(cache / 'site'), str(cache / 'ml-sharp' / 'src')]
compiled = next((cache / 'extensions').rglob('gsplat_cuda.so'))
spec = importlib.util.spec_from_file_location('gsplat_cuda', compiled)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
sys.modules['gsplat.csrc'] = native
from gsplat import rasterization
from sharp.utils.color_space import sRGB2linearRGB, linearRGB2sRGB

payload = next(ROOT.rglob('payload.zip'), None)
project = WORK / 'input'
if payload:
    with zipfile.ZipFile(payload) as bundle:
        for member in bundle.infolist():
            if not (project / member.filename).resolve().is_relative_to(project.resolve()):
                raise ValueError('Unsafe archive path')
        bundle.extractall(project)
else:
    project = next(ROOT.rglob('approach-jobs.json')).parent
jobs = json.loads((project / 'approach-jobs.json').read_text())
device = torch.device('cuda')


def tensor(array):
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy()).to(device)


def guide(src, dst, plan):
    dtype = np.dtype([('pos','<f4',3),('scale','<f4',3),('rgba','u1',4),('quat','u1',4)])
    records = np.fromfile(src / 'base.splat', dtype=dtype)
    means, scales = tensor(records['pos']), tensor(records['scale'])
    quats = tensor((records['quat'].astype(float)-128)/128)
    quats /= quats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    colors = sRGB2linearRGB(tensor(records['rgba'][:,:3]/255))
    opacity = tensor(records['rgba'][:,3]/255)
    # Road coordinates are x right/y up/z forward; .splat files use z back.
    camera = np.array([0., plan['camera_height_m'], plan['distance_m']])
    rotation = np.diag([1.,1.,-1.]) @ np.array(plan['camera_to_road'])
    view = np.eye(4)
    view[:3,:3] = rotation.T
    view[:3,3] = -rotation.T @ camera
    K = tensor([[[plan['focal_px'],0,plan['width']/2],
                 [0,plan['focal_px'],plan['height']/2],[0,0,1]]])
    views = tensor(view[None])
    with torch.inference_mode():
        original, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opacity,
            colors=colors, viewmats=views, Ks=K, width=plan['width'], height=plan['height'],
            packed=True, render_mode='RGB+ED')
    # Geometry-only scaffold for the unseen asphalt. This is a guide, not the delivered photograph.
    xs = np.arange(-8, 16, .22)
    zs = np.arange(-plan['distance_m']+.2, 4, .22)
    xx, zz = np.meshgrid(xs,zs)
    points = np.column_stack([xx.ravel(), np.zeros(xx.size), -zz.ravel()])
    rgb = np.repeat(np.array(plan['ground_rgb'])[None], len(points), axis=0)
    road = (points[:,0]>=-1.75)&(points[:,0]<=8.75)
    rgb[~road] *= .7
    lane = np.minimum(abs(points[:,0]-1.75),abs(points[:,0]-5.25))<.14
    dashed = np.mod(-points[:,2],9)<3
    edges = (abs(points[:,0]+1.75)<.14)|(abs(points[:,0]-8.75)<.14)
    rgb[(lane&dashed)|edges] = [.72,.72,.68]
    road_means = tensor(points)
    road_quats = tensor(np.tile([1.,0,0,0],(len(points),1)))
    road_scales = tensor(np.tile([.18,.008,.18],(len(points),1)))
    road_colors = sRGB2linearRGB(tensor(rgb))
    with torch.inference_mode():
        fill, _, _ = rasterization(means=road_means, quats=road_quats, scales=road_scales,
            opacities=torch.full((len(points),),.99,device=device), colors=road_colors,
            viewmats=views,Ks=K,width=plan['width'],height=plan['height'],packed=True)
    linear = original[0,:,:,:3] + fill[0]*(1-alpha[0])
    rgb = linearRGB2sRGB(linear).clamp(0,1).cpu().numpy()
    known = (alpha[0,:,:,0]>.995).cpu().numpy()
    Image.fromarray((rgb*255).astype(np.uint8)).save(dst/'guide.png')
    Image.fromarray((known*255).astype(np.uint8)).save(dst/'known.png')
    np.savez_compressed(dst/'guide-depth.npz', depth=original[0,:,:,3].cpu().numpy().astype(np.float16),
                        alpha=alpha[0,:,:,0].cpu().numpy().astype(np.float16))
    return {'known_fraction':float(known.mean()), 'phase':'guide'}


predictor = None
if PHASE == 'predict':
    from sharp.models import PredictorParams, create_predictor
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import save_ply
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(torch.load(cache/'sharp.pt',map_location='cpu',weights_only=True))
    predictor.eval().cuda()

for plan in jobs:
    src = project / plan['case']
    dst = WORK / 'approach' / plan['case']
    dst.mkdir(parents=True,exist_ok=True)
    start = time.time()
    if PHASE == 'guide':
        extra = guide(src,dst,plan)
    elif PHASE == 'predict':
        image_path = src/'extended.png'
        if hashlib.sha256(image_path.read_bytes()).hexdigest()!=plan['image_sha256']:
            raise ValueError('Reviewed extension image hash mismatch')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        with torch.inference_mode():
            result = predict_image(predictor,image,plan['focal_px'],device)
        save_ply(result,plan['focal_px'],image.shape[:2],dst/'approach.ply')
        del result
        extra = {'phase':'predict','image_sha256':plan['image_sha256']}
    else:
        raise ValueError(PHASE)
    (dst/'result.json').write_text(json.dumps(dict(plan,**extra,seconds=round(time.time()-start,2),
        gpu=torch.cuda.get_device_name(0),runtime=runtime),indent=2))
    torch.cuda.empty_cache()
    print('DONE',PHASE,plan['case'],flush=True)
