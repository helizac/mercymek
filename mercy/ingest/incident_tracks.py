"""Preserve visually identified collision parties when sparse neural-depth sampling misses them.

These tracks are explicitly estimated from original boxes, vehicle-class dimensions and the shared
road/camera calibration. They are not depth measurements or newly invented vehicle detections.
"""
import json

import numpy as np

from mercy.generation.vehicles import DIMENSIONS, vehicle_observations
from mercy.ingest.vehicle_scene import Cameras, cuboid, fit_box


def ensure(case):
    path=case/'tracks.json'
    tracks=json.loads(path.read_text())
    scene=json.loads((case/'scene.json').read_text())
    parties={v['label']:v for v in scene['vehicles'] if v['role'] in {'striking','struck'}}
    if not parties:return tracks
    meta=json.loads((case/'meta.json').read_text())
    impact=float(scene.get('impact_time_s') or meta['time_of_event'])
    overlays=json.loads((case/'overlays.json').read_text())
    observations=vehicle_observations(overlays)
    aliases=tracks.get('aliases',{})
    bindings={}
    for label in parties:
        candidates=[key for key,value in aliases.items() if value==label and key in observations]
        if not candidates:continue
        def quality(key):
            nearest=min(abs(f/meta['src_fps']-impact) for f,_ in observations[key])
            area=max((o['box'][2]-o['box'][0])*(o['box'][3]-o['box'][1])
                     for f,o in observations[key] if abs(f/meta['src_fps']-impact)<=nearest+.1)
            return nearest,-area
        key=min(candidates,key=quality)
        if quality(key)[0]<=1.5:bindings[label]=key
    needs=[key for key in bindings.values() if sum(abs(r['t']-impact)<=1.2 for r in tracks['tracks'].get(key,[]))<3
           and key not in tracks.get('track_seeds',{})]
    if needs:
        frames=json.loads((case/'frames4d.json').read_text())['frames']
        with np.load(case/'recon.npz') as rec:
            times=rec['times'];K=rec['intrinsic'].copy();height,width=rec['depth'].shape[1:]
        K*=np.array([[meta['width']/width],[meta['height']/height],[1.]])
        cameras=Cameras(times,np.array([f['M'] for f in frames]),K)
        for key in needs:
            label=next(label for label,value in bindings.items() if value==key)
            kind=parties[label]['body_type'].lower().replace(' ','_')
            kind={'car':'sedan','truck':'box_truck','motorbike':'motorcycle'}.get(kind,kind)
            dimensions=DIMENSIONS.get(kind,DIMENSIONS['sedan'])
            rows=[]
            for frame,obj in observations[key]:
                t=frame/meta['src_fps'];C,R,lens=cameras.at(t)
                yaw=float(np.degrees(np.arctan2(R[0,2],R[2,2])))
                xz,yaw,error=fit_box(obj['box'],dimensions,C,R,lens,yaw,(meta['width'],meta['height']))
                if not np.isfinite(xz).all():continue
                body=cuboid(dimensions,yaw)+[xz[0],0,xz[1]]
                rows.append({'k':int(np.argmin(abs(times-t))),'t':round(t,6),
                             'center':[float(xz[0]),dimensions[1]/2,float(xz[1])],
                             'bbox':[body.min(0).tolist(),body.max(0).tolist()],
                             'n':0,'method':'source-box class-size road-plane estimate','source_frame':frame,
                             'source_box_error_px':float(error)})
            if len(rows)<3:continue
            tracks['tracks'][key]=rows
            tracks.setdefault('track_seeds',{})[key]={'method':'original source boxes + vehicle-class dimensions + shared road plane',
                'metric_depth_measured':False,'dimensions_m':dimensions,'observations':len(rows)}
    # One observed physical candidate for each semantic incident party. Other vehicles remain in the
    # inventory; a similarly colored bystander must not become the second collision participant.
    for key,label in list(aliases.items()):
        if label in bindings and key!=bindings[label]:
            tracks.setdefault('unselected_incident_aliases',{})[key]=aliases.pop(key)
    tracks['incident_bindings']=bindings
    if not (case/'tracks.neural-original.json').exists():
        (case/'tracks.neural-original.json').write_text(path.read_text())
    path.write_text(json.dumps(tracks,separators=(',',':')))
    return tracks
