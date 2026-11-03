"""Plan and fuse a generated approach corridor, preserving the original world and observed motion."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from mercy.ingest.environment import read_gaussians, transform_gaussians, write_splat


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(case):
    env = case/'environment'
    reconstruction = json.loads((env/'reconstruction.json').read_text())
    moments = json.loads((case/'moments.json').read_text())
    moment = next((m for m in moments['moments'] if m['id']=='impact'),moments['moments'][-1])
    keys = [key for key in moment['camera_keys'] if moment['replay_start']<=key[0]<=moment['replay_end']]
    if not keys:
        raise ValueError('No observed camera trajectory in the replay interval')
    distance = min(160.,math.ceil((max(20.,-min(key[2] for key in keys))+12)/5)*5)
    image = cv2.imread(str(env/'empty.png'))
    h,w = image.shape[:2]
    road = reconstruction['road']
    result = {'case':case.name,'distance_m':distance,'width':1280,'height':round(h*1280/w),
              'focal_px':reconstruction['focal_px']*1280/w,
              'camera_height_m':moment['cam']['pos'][1], 'camera_to_road':road['basis'],
              'ground_rgb':moment['cam']['ground'], 'base_world_sha256':digest(env/'world.splat'),
              'source_image_sha256':digest(env/'empty.png'),'generated':True,
              'lane_scaffold':'three 3.5m lanes, camera in left lane; approximate guide, refined by Muse',
              'version':'backwards-fixed-camera-road-v1'}
    result['plan_id'] = hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def fuse(case, plan, output):
    root = case/'environment'/'approach'
    review = json.loads((root/'review.json').read_text())
    result = json.loads((output/'result.json').read_text())
    if not review.get('accepted') or review['image_sha256']!=digest(root/'extended.png'):
        raise ValueError('Road extension image lacks a valid accepted review')
    if result['image_sha256']!=review['image_sha256'] or result['plan_id']!=plan['plan_id']:
        raise ValueError('GPU extension does not match its image and camera plan')
    base = case/'environment'/'world.splat'
    if digest(base)!=plan['base_world_sha256']:
        raise ValueError('The original world changed during extension')
    g = read_gaussians(output/'approach.ply')
    keep = np.isfinite(g['pos']).all(1)&np.isfinite(g['scales']).all(1)&(g['pos'][:,2]>.3)&(g['op']>.035)
    g = {key:value[keep] for key,value in g.items()}
    P = g['pos']
    width,height,focal = plan['width'],plan['height'],plan['focal_px']
    u,v = focal*P[:,0]/P[:,2]+width/2,focal*P[:,1]/P[:,2]+height/2
    inb = (u>=0)&(u<width)&(v>=0)&(v<height)
    ui,vi = np.clip(u,0,width-1).astype(int),np.clip(v,0,height-1).astype(int)
    depth = np.load(root/'guide-depth.npz')
    reference = depth['depth'][vi,ui].astype(float)
    known = (depth['alpha'][vi,ui]>.995)&inb&(reference>1)&(reference<150)
    protected = cv2.imread(str(root/'protected.png'),cv2.IMREAD_GRAYSCALE) if (root/'protected.png').exists() else None
    if protected is not None:
        known &= protected[vi,ui]>0
    ratios = reference[known]/P[known,2]
    ratios = ratios[np.isfinite(ratios)&(ratios>.05)&(ratios<20)]
    basis = np.array(plan['camera_to_road'])
    road_mask = np.zeros((height,width),np.uint8)
    polygon = np.array(review['road_polygon'],float)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or not 4<=len(polygon)<=24 or (polygon<0).any() or (polygon>1000).any():
        raise ValueError('Invalid reviewed roadway polygon')
    cv2.fillPoly(road_mask,[(polygon/1000*[width-1,height-1]).astype(np.int32)],1)
    on_road = inb&(road_mask[vi,ui]>0)
    if len(ratios)>=200 and np.median(np.abs(ratios-np.median(ratios)))/np.median(ratios)<.35:
        scale = float(np.median(ratios))
        alignment = {'method':'robust depth ratio on preserved original-world overlap','observations':len(ratios),
                     'relative_mad':float(np.median(np.abs(ratios-scale))/scale)}
    else:
        offsets = P[on_road] @ (-basis[1])
        offsets = offsets[(offsets>.15)&(offsets<15)]
        if len(offsets)<100:
            raise ValueError('Neither overlap nor road plane supports extension alignment')
        scale = plan['camera_height_m']/float(np.median(offsets))
        alignment = {'method':'explicit camera-height/planar-road prior; overlap insufficient','observations':len(offsets)}
    # One shared ground plane prevents disconnected independently predicted road slabs.
    rays = P/P[:,2:]
    vertical = rays @ basis[1]
    on_road &= vertical<-.015
    corrected = -plan['camera_height_m']/vertical[on_road]/scale
    ratio = corrected/P[on_road,2]
    g['pos'][on_road] = rays[on_road]*corrected[:,None]
    g['scales'][on_road] *= ratio[:,None]
    pos,scales,quat = transform_gaussians(g,basis,scale,plan['camera_height_m'])
    # transform_gaussians produces browser axes, where backwards is +z.
    pos[:,2] += plan['distance_m']
    road_z = -pos[:,2]
    selected = (road_z<3)&(road_z>-plan['distance_m']+.4)&(abs(pos[:,0])<55)&(pos[:,1]>-.4)&(pos[:,1]<35)
    selected &= np.isfinite(scales).all(1)&(scales.max(1)<4)
    idx = np.flatnonzero(selected)
    if len(idx)<5000:
        raise ValueError('Insufficient generated geometry behind the original world')
    if len(idx)>1_000_000:
        idx=idx[np.linspace(0,len(idx)-1,1_000_000).astype(int)]
    opacity = g['op'][idx]*np.clip((3-road_z[idx])/1.5,0,1)
    extension = root/'extension.splat'
    write_splat(extension,pos[idx],scales[idx],quat[idx],g['rgb'][idx],opacity)
    combined = case/'environment'/'extended-world.splat'
    temp = combined.with_suffix('.tmp')
    with temp.open('wb') as stream:
        stream.write(base.read_bytes())  # Preserve every original Gaussian byte-for-byte.
        stream.write(extension.read_bytes())
    temp.replace(combined)
    info = {**plan,'alignment':dict(alignment,scale=scale),'extension_splats':len(idx),
            'image_sha256':review['image_sha256'],'extended_world_sha256':digest(combined),
            'road_constraint':'one shared y=0 plane; height and unseen surroundings estimated',
            'file':'environment/extended-world.splat','image':'environment/approach/extended.png',
            'bounds':np.array([pos[idx].min(0),pos[idx].max(0)]).tolist(),
            'source_trajectory':'unchanged recorded reconstruction keys; no invented pre-roll motion'}
    (root/'extension.json').write_text(json.dumps(info,indent=2))
    moments_path = case/'moments.json'
    manifest = json.loads(moments_path.read_text())
    for moment in manifest['moments']:
        moment['base_file']='environment/world.splat'
        moment['file']=info['file']
        moment['count']=combined.stat().st_size//32
        moment['approach']={'distance_m':plan['distance_m'],'generated':True,'camera_height_m':plan['camera_height_m'],
                            'plan_id':plan['plan_id'],'image':info['image'],'road_z_min':-plan['distance_m']}
    manifest['approach']={'generated':True,'plan_id':plan['plan_id'],'distance_m':plan['distance_m']}
    temporary = moments_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest,indent=1));temporary.replace(moments_path)
    return info
