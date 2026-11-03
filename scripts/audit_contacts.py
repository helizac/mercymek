"""Independently validate published/candidate interpolation and source-edge reprojection."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mercy.ingest.vehicle_scene import Cameras, continuous_cameras, fitting_points, observed_edges, project_fitting_points
from mercy.ingest.playback_motion import camera_path
from mercy.ingest.traffic_constraints import body_dimensions, swept_conflicts, source_quality


def audit(name, candidate=False):
    case=ROOT/'data/cases'/name
    meta=json.loads((case/'meta.json').read_text());manifest=json.loads((case/'moments.json').read_text())
    m=manifest['moments'][0]
    with np.load(case/'recon.npz') as rec:
        times=rec['times'];K=rec['intrinsic'].copy();rh,rw=rec['depth'].shape[1:]
    K*=np.array([[meta['width']/rw],[meta['height']/rh],[1.]])
    with np.load(case/'environment/cameras.npz') as rec:matrices=rec['matrices']
    matrices=continuous_cameras(times,matrices,m['replay_start'],m['replay_end'])
    matrices,velocity=camera_path(case,times,matrices,json.loads((case/'environment/input.json').read_text())['anchor_t'],return_velocity=True,write_report=False)
    cameras=Cameras(times,matrices,K,position_velocity=velocity)
    projection=[]
    for target in [3.54,1.02,5.11,2.27]:
        t=round(np.floor(target*meta['src_fps']+1e-6)/meta['src_fps'],6)
        C,R,lens=cameras.at(t)
        cv=np.array([[.4,-.2,8],[-1.,.3,12],[.2,.8,4.]])
        points=cv@R.T+C
        pixels=cv[:,:2]/cv[:,2,None]*[lens[0,0],lens[1,1]]+[lens[0,2],lens[1,2]]
        projection.append({'time':t,'points_road':points.tolist(),'pixels':pixels.tolist()})
    (ROOT/'data/research/contacts'/f'{name}-projection.json').write_text(json.dumps(projection))
    if candidate:templates=json.loads((ROOT/'data/research/contacts'/f'{name}-candidate.json').read_text())['templates']
    else:templates=[{**v,'keys_absolute':[[k[0]+m['t_frame'],*k[1:]] for k in v['keys']]} for v in m['vehicles']]
    before=json.loads((ROOT/'data/research/contacts/before'/name/'moments.json').read_text())['moments'][0]
    old={v['key']:np.array([[k[0]+before['t_frame'],*k[1:]] for k in v['keys']]) for v in before['vehicles']}
    keys=np.array([v['keys_absolute'] for v in templates]);dims=np.array([body_dimensions(v['model']) for v in templates])
    conflicts=swept_conflicts(keys[:,:,1:3],keys[:,:,3],dims)
    previous=np.array([old[v['key']] for v in templates])
    before_conflicts=swept_conflicts(previous[:,:,1:3],previous[:,:,3],dims)
    overlays=json.loads((case/'overlays.json').read_text());scores={}
    for item in templates:
        key=item['key'];new=np.array(item['keys_absolute']);sourcekeys=item['model'].get('source_keys',[key]);errors=[[],[]]
        for f,objects in overlays.items():
            t=int(f)/meta['src_fps'];choices=[(k,objects[k]) for k in sourcekeys if k in objects]
            if not choices:continue
            sourcekey,obj=max(choices,key=lambda x:x[1].get('confidence',0))
            if not source_quality(obj,(meta['width'],meta['height'])):continue
            edges=observed_edges(sourcekey,obj,objects,(meta['width'],meta['height']))
            if not edges.any():continue
            box=np.array(obj['box']);scale=np.array([box[2]-box[0],box[3]-box[1]]*2).clip(6)
            for j,k in enumerate([old[key],new]):
                x,z,yaw=[np.interp(t,k[:,0],k[:,axis]) for axis in [1,2,3]]
                clip_x=(0,meta['width']) if box[0]<=2 or box[2]>=meta['width']-3 else None
                predicted=project_fitting_points(fitting_points(item['model'],yaw),[x,z],*cameras.at(t),clip_x)
                errors[j].extend((abs(predicted-box)/scale)[edges])
        if errors[0]:scores[key]={'edges':len(errors[0]),'before_median':float(np.median(errors[0])),
                                'after_median':float(np.median(errors[1])),'after_p90':float(np.quantile(errors[1],.9))}
    return {'case':name,'continuous_conflicting_intervals':len(conflicts),'before_conflicting_intervals':len(before_conflicts),
            'actors':len(templates),'source_edge_errors_in_box_extents':scores}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',nargs='+',required=True);p.add_argument('--candidate',action='store_true');args=p.parse_args()
    reports=[audit(name,args.candidate) for name in args.case]
    (ROOT/'data/research/contacts/audit.json').write_text(json.dumps(reports,indent=2))
    for report in reports:print(json.dumps(report,indent=2))
