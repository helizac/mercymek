"""Video-supported exceptions to the default straight-lane traffic prior."""
import hashlib
import json
import re

import cv2

from mercy import config
from mercy.meta_client import structured, data_url, upload_file
from mercy.generation.vehicles import vehicle_observations

SCHEMA={'type':'object','additionalProperties':False,'required':['exceptions','notes'],'properties':{
 'notes':{'type':'string'},'exceptions':{'type':'array','items':{'type':'object','additionalProperties':False,
 'required':['track','behavior','confidence','evidence'],'properties':{'track':{'type':'string'},
 'behavior':{'type':'string','enum':['stationary','opposite_straight','lane_change','turn','crossing','skidding','uncertain']},
 'confidence':{'type':'number'},'evidence':{'type':'string'}}}}}}


def review(case):
    dst=case/'vehicles';dst.mkdir(exist_ok=True)
    target=dst/'motion-review.json'
    signature=hashlib.sha256((case/'clip.mp4').read_bytes()+(case/'overlays.json').read_bytes()+b'straight-traffic-exceptions-v1').hexdigest()
    if target.exists():
        cached=json.loads(target.read_text())
        if cached.get('signature')==signature:return cached
    meta=json.loads((case/'meta.json').read_text());physics=json.loads((case/'physics.json').read_text())
    overlays=json.loads((case/'overlays.json').read_text());observations=vehicle_observations(overlays)
    primary=[o['id'] for o in physics['objects'] if o.get('vid')]
    files=config.ROOT/'data/research/motion-inputs';files.mkdir(parents=True,exist_ok=True)
    remote=files/(hashlib.sha256((case/'clip.mp4').read_bytes()).hexdigest()+'.json')
    if remote.exists():fid=json.loads(remote.read_text())['file_id']
    else:
        fid=upload_file(case/'clip.mp4');remote.write_text(json.dumps({'file_id':fid}))
    content=[{'type':'input_text','text':
      'Watch the ORIGINAL dashcam clip and distinguish body/world motion from motion CAUSED BY THE CAMERA. '
      'Default for normal traffic is driving STRAIGHT FORWARD in the same lane, possibly slowing/stopping. '
      'We need only clearly evidenced exceptions to this default, and truly parked/stationary vehicles. '
      'A car showing its side because it is alongside the camera is NOT crossing or turning. Slanted screen '
      'motion/perspective, a noisy box, occlusion, or a single sideways-looking crop is NOT a lane change. '
      'The primary incident participants '+json.dumps(primary)+' already have flexible trajectories. '
      'Use the annotated source frames to identify track IDs; the full video is the motion evidence. '
      'For each clear exception name a supplied track, one behavior, confidence 0..1, and specific visible '
      'temporal evidence. Only identify opposing traffic if its front and road direction support it. '
      'A brief stop in a queue is still straight traffic, not stationary for the whole clip. '
      'Do not invent maneuvers; mark uncertain if no clear temporal evidence. Allowed track IDs: '+json.dumps(list(observations))},
      {'type':'input_file','file_id':fid}]
    cap=cv2.VideoCapture(str(case/'clip.mp4'))
    try:
        t0=float(physics['impact']['t'])
        for t in [max(0,t0-2.5),max(0,t0-.6),min(meta['duration']-.1,t0+1.0)]:
            f=round(t*meta['src_fps']);cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read()
            if not ok:continue
            for key,obj in overlays.get(str(f),{}).items():
                if key not in observations:continue
                x1,y1,x2,y2=map(int,obj['box']);cv2.rectangle(image,(x1,y1),(x2,y2),(20,240,120),1)
                cv2.putText(image,key,(x1,max(14,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,30,240),1,cv2.LINE_AA)
            path=dst/f'motion-context-{f:04d}.jpg';cv2.imwrite(str(path),image)
            content.extend([{'type':'input_text','text':f'Original source frame {f}, t={t:.2f}s, track IDs overlaid'},
                            {'type':'input_image','image_url':data_url(path)}])
    finally:cap.release()
    result=structured(config.VISION_MODEL,[{'role':'user','content':content}],SCHEMA,'traffic_motion_exceptions',effort='high',timeout=240)
    if any(x['track'] not in observations for x in result['exceptions']):raise ValueError('Motion review referenced an unknown source track')
    result.update(signature=signature,primary=primary,default='straight_forward',model=config.VISION_MODEL)
    target.write_text(json.dumps(result,indent=2))
    print(case.name,'motion behavior reviewed',len(result['exceptions']),'exceptions',flush=True)
    return result
