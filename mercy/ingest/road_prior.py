"""Explicit planar-road prior for wet scenes where monocular reflection depth is unsupported."""
import json

import cv2
import numpy as np

from mercy import config
from mercy.meta_client import data_url, structured

SCHEMA = {"type":"object", "additionalProperties":False, "required":["polygon", "notes"], "properties":{
    "polygon":{"type":"array", "items":{"type":"array", "items":{"type":"integer"}}},
    "notes":{"type":"string"}}}


def apply(case, g, result, camera_rotation, width, height, reason):
    path = case / 'environment' / 'road-mask.json'
    sha = result['image_sha256']
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get('image_sha256') != sha:
        saved = structured(config.VISION_MODEL, [{"role":"user", "content":[
            {"type":"input_text", "text":"Trace the visible drivable asphalt ROAD SURFACE in this EMPTY scene as one "
             "clockwise polygon of 4 to 24 [x,y] vertices, normalized 0..1000 for both axes. Follow the road edges "
             "and horizon. Exclude buildings, sky, fences, poles and sidewalks. Include wet road reflections as road. "
             "This is a semantic surface mask, not a depth estimate. Give short notes."},
            {"type":"input_image", "image_url":data_url(case / 'environment' / 'empty.png')}]}],
            SCHEMA, 'road_surface', timeout=90)
        saved.update(image_sha256=sha, model=config.VISION_MODEL)
        path.write_text(json.dumps(saved,indent=2))
    polygon = np.asarray(saved['polygon'], float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not 4 <= len(polygon) <= 24 or (polygon < 0).any() or (polygon > 1000).any():
        raise ValueError('Invalid road polygon')
    mask = np.zeros((height,width),np.uint8)
    cv2.fillPoly(mask,[(polygon/1000*[width-1,height-1]).astype(np.int32)],1)
    if mask.mean() < .10 or mask.mean() > .8:
        raise ValueError('Road mask has implausible coverage')
    P = g['pos']
    u = result['focal_px']*P[:,0]/P[:,2]+width/2
    v = result['focal_px']*P[:,1]/P[:,2]+height/2
    inb = (u>=0)&(u<width)&(v>=0)&(v<height)
    selected = np.zeros(len(P),bool)
    selected[inb] = mask[v[inb].astype(int),u[inb].astype(int)] > 0
    n = -np.asarray(camera_rotation)[1,:]
    n /= np.linalg.norm(n)
    if n[1] < .85:
        raise ValueError('Prior camera has unsupported road orientation')
    offsets = P[selected] @ n
    offset = float(np.median(offsets[(offsets>.15)&(offsets<8)]))
    if not np.isfinite(offset):
        raise ValueError('No usable road depth to anchor the explicit plane prior')
    rays = P / P[:,2:]
    denominator = rays @ n
    selected &= denominator > .03
    depth = offset / np.maximum(denominator[selected],.03)
    ratio = depth/P[selected,2]
    g['pos'][selected] = rays[selected]*depth[:,None]
    g['scales'][selected] *= ratio[:,None]
    forward = np.array([0.,0.,1.])-n*n[2]; forward/=np.linalg.norm(forward)
    basis = np.stack([np.cross(n,forward),-n,forward])
    road = {'basis':basis.tolist(), 'normal_camera':n.tolist(), 'offset':offset, 'scale':1.35/offset,
            'inlier_ratio':None, 'residual_m_unscaled':None,
            'scale_method':'assumed 1.35m camera height, planar road prior; wet-road monocular depth rejected',
            'method':'explicit planar road prior', 'reason':reason, 'road_mask':str(path.name),
            'corrected_gaussians':int(selected.sum())}
    return g, road
