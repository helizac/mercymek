"""Review incident body headings in full source videos, explicitly separating perspective and turning."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from mercy import config
from mercy.meta_client import structured, data_url, upload_file
ROOT = config.ROOT

SCHEMA = {'type':'object','additionalProperties':False,'required':['headings','notes'], 'properties':{
    'notes':{'type':'string'}, 'headings':{'type':'array','items':{'type':'object','additionalProperties':False,
        'required':['track','source_frame','road_yaw_deg','confidence','evidence'], 'properties':{
            'track':{'type':'string'},'source_frame':{'type':'integer'},'road_yaw_deg':{'type':'number'},
            'confidence':{'type':'number'},'evidence':{'type':'string'}}}}}}


def review(name):
    case = config.case_dir(name)
    meta = json.loads((case/'meta.json').read_text())
    first = json.loads((case/'moments.json').read_text())['moments'][0]
    physics = json.loads((case/'physics.json').read_text())
    ids = [o['id'] for o in physics['objects'] if o.get('vid') and o['id'] != 'ego']
    overlays = json.loads((case/'overlays.json').read_text())
    signature = hashlib.sha256((case/'clip.mp4').read_bytes()+(case/'overlays.json').read_bytes()+b'road-relative-video-heading-v1').hexdigest()
    target = case/'vehicles/heading-review.json'
    if target.exists() and json.loads(target.read_text()).get('signature') == signature:
        return json.loads(target.read_text())
    remote = ROOT/'data/research/motion-inputs'/ (hashlib.sha256((case/'clip.mp4').read_bytes()).hexdigest()+'.json')
    if remote.exists():
        fid = json.loads(remote.read_text())['file_id']
    else:
        fid = upload_file(case/'clip.mp4'); remote.parent.mkdir(parents=True,exist_ok=True)
        remote.write_text(json.dumps({'file_id':fid}))
    content = [{'type':'input_text','text':
        'Check the BODY NOSE heading of the specified incident vehicles throughout this ORIGINAL video. '
        'Previous single-crop estimates confused SIDE PERSPECTIVE with a 90-degree turn and sometimes '
        'confused left with right. Determine the road-relative heading, using lanes, front/rear lights and '
        'the whole video. 0 degrees = along the camera vehicle lane FORWARD, 180 = opposite traffic, '
        '+90 = perpendicular across the road towards IMAGE RIGHT, -90 = towards IMAGE LEFT. '
        'A vehicle alongside on the right can show its entire LEFT flank with its front on the LEFT of '
        'the crop while STILL pointing along the road (near 0 degrees). Do NOT call that -90. '
        'A car turning across an intersection CAN change continuously from opposite traffic to +90 or -90. '
        'Screen translation alone is not a world heading. Look at the front/rear relative to the ROAD, '
        'not relative to the ray from the camera to the car. Do not assume earlier crop descriptions are right. '
        'For each supplied annotated frame and target track, provide a conservative road_yaw_deg in [-180,180] '
        'with confidence 0..1 and short physical visual evidence. If hidden, mark low confidence; do not invent. '
        'Targets: '+json.dumps(ids)}, {'type':'input_file','file_id':fid}]
    cap = cv2.VideoCapture(str(case/'clip.mp4')); supplied=set()
    try:
        for t in np.linspace(first['replay_start'],first['replay_end'],7):
            frame = round(float(t)*meta['src_fps']); cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
            if not ok:continue
            present=[]
            for key in ids:
                obj=overlays.get(str(frame),{}).get(key)
                if not obj:continue
                present.append(key);x1,y1,x2,y2=map(int,obj['box']);cv2.rectangle(image,(x1,y1),(x2,y2),(20,230,255),3)
                cv2.putText(image,key,(max(0,x1),max(20,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,230,255),2)
            if not present:continue
            image=cv2.resize(image,(960,round(image.shape[0]*960/image.shape[1])))
            path=case/'vehicles'/f'heading-context-{frame:04d}.jpg';cv2.imwrite(str(path),image)
            supplied.add(frame)
            content += [{'type':'input_text','text':f'source_frame={frame}, t={t:.2f}, targets={present}'},
                        {'type':'input_image','image_url':data_url(path)}]
    finally:cap.release()
    result=structured(config.VISION_MODEL,[{'role':'user','content':content}],SCHEMA,'road_relative_body_heading',effort='high',timeout=240)
    if any(h['track'] not in ids or h['source_frame'] not in supplied or not -180<=h['road_yaw_deg']<=180 for h in result['headings']):
        raise ValueError('Heading review referenced an unsupported source frame/track')
    result.update(signature=signature,model=config.VISION_MODEL,reference='road +Z forward, +X right; perspective accounted for')
    target.write_text(json.dumps(result,indent=2))
    print(name,result,flush=True)
    return result


