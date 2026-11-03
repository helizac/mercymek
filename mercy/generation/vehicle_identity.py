"""Conservative source-image re-identification across detector ID changes; cache every decision."""
import hashlib
import json
import re

import numpy as np

from mercy import config
from mercy.meta_client import data_url, structured, has_key

SCHEMA={'type':'object','additionalProperties':False,'required':['pairs'],'properties':{'pairs':{'type':'array','items':{
 'type':'object','additionalProperties':False,'required':['earlier','later','same_vehicle','confidence','evidence'],
 'properties':{'earlier':{'type':'string'},'later':{'type':'string'},'same_vehicle':{'type':'boolean'},
               'confidence':{'type':'number'},'evidence':{'type':'string'}}}}}}


def signature(case,profiles):
    return hashlib.sha256(((case/'overlays.json').read_bytes()+json.dumps({k:v.get('signature') for k,v in profiles.items()},sort_keys=True).encode()+b'identity-v1')).hexdigest()


def review(case,profiles,observations):
    path=case/'vehicles/identity.json';sig=signature(case,profiles)
    if path.exists():
        cached=json.loads(path.read_text())
        if cached.get('signature')==sig:return cached
    meta=json.loads((case/'meta.json').read_text());W,H=meta['width'],meta['height'];fps=meta['src_fps']
    candidates=[]
    for a,ra in observations.items():
        for b,rb in observations.items():
            gap=(rb[0][0]-ra[-1][0])/fps
            if not .05<gap<3.5:continue
            A,B=ra[-1][1]['box'],rb[0][1]['box']
            delta=np.array([(A[0]+A[2]-B[0]-B[2])/(2*W),(A[1]+A[3]-B[1]-B[3])/(2*H)])
            if abs(delta[0])>.28 or abs(delta[1])>.25:continue
            if max((r[1]['box'][2]-r[1]['box'][0])*(r[1]['box'][3]-r[1]['box'][1]) for r in ra)<600:continue
            pa,pb=profiles[a].get('plate',{}),profiles[b].get('plate',{})
            if pa.get('observed') and pb.get('observed') and pa['text']!=pb['text']:continue
            candidates.append((float(np.linalg.norm(delta)),a,b))
    # At most the two nearest plausible continuations per disappearing track.
    chosen=[];counts={}
    for _,a,b in sorted(candidates):
        if counts.get(a,0)>=2:continue
        chosen.append((a,b));counts[a]=counts.get(a,0)+1
    result={'signature':sig,'method':'original-crop/context identity review; high-confidence links only',
            'pairs':[],'accepted':[]}
    if not chosen or not has_key():
        path.write_text(json.dumps(result,indent=2));return result
    for offset in range(0,len(chosen),4):
        batch=chosen[offset:offset+4];keys=sorted({k for pair in batch for k in pair})
        content=[{'type':'input_text','text':
          'Check whether detector IDs before/after an occlusion belong to the SAME physical car. '
          'Candidate (earlier,later) pairs: '+json.dumps(batch)+'. Compare actual body silhouette, lamp shapes, '
          'paint, windows and source-frame traffic context. A generic similar colour/type is NOT sufficient. '
          'The camera moves forward. Cars may disappear behind a nearer car and acquire another ID. '
          'Return every candidate pair; same_vehicle=false with low confidence when identity is ambiguous. '
          'A positive decision needs specific consistent visual details or an unambiguous occlusion path. '
          'Never read/invent plate characters. The outlines indicate the intended vehicle, not an occluder.'}]
        for key in keys:
            stem=re.sub(r'[^\w-]','_',key)
            paths=sorted((case/'vehicles').glob(stem+'.f*.png'))
            earlier=any(a==key for a,b in batch)
            path_crop=paths[-1 if earlier else 0]
            context=path_crop.with_name(path_crop.stem+'.context.jpg')
            content += [{'type':'input_text','text':f'{key}, {path_crop.stem}, type estimate {profiles[key]["body_type"]}'},
                        {'type':'input_image','image_url':data_url(path_crop)},
                        {'type':'input_image','image_url':data_url(context)}]
        answer=structured(config.VISION_MODEL,[{'role':'user','content':content}],SCHEMA,'vehicle_identity',timeout=150)
        for pair in answer['pairs']:
            if (pair['earlier'],pair['later']) not in batch:raise ValueError('Identity review referenced an unrequested track pair')
            result['pairs'].append(pair)
        print(case.name,'identity pairs checked',len(result['pairs']),flush=True)
    used_before=set();used_after=set()
    for pair in sorted(result['pairs'],key=lambda p:-p['confidence']):
        a,b=pair['earlier'],pair['later']
        if pair['same_vehicle'] and pair['confidence']>=.90 and a not in used_before and b not in used_after:
            result['accepted'].append(pair);used_before.add(a);used_after.add(b)
    path.write_text(json.dumps(result,indent=2));return result
