"""Compare saved RGB-only detections to source SAM boxes and screen collapsed cuboids.

This is a coverage/structural sanity check, not a 3D accuracy benchmark: no 3D ground truth is available.
"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mercy import config
from mercy.generation.vehicles import vehicle_observations


def iou(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))


def evaluate():
    runs=[];sources={}
    for file in sorted((ROOT/'data/kaggle-vehicle-pose/output').glob('*/pose-predictions/results.json')):
        rows=json.loads(file.read_text());frames=[]
        for row in rows:
            name=row['case']
            if name not in sources:
                overlays=json.loads((config.case_dir(name)/'overlays.json').read_text())
                sources[name]=(overlays,vehicle_observations(overlays))
            overlays,tracks=sources[name]
            refs=[(k,v) for k,v in overlays.get(str(row['frame']),{}).items() if k in tracks and
                  v['box'][2]-v['box'][0]>=7 and v['box'][3]-v['box'][1]>=6]
            pred=row['predictions'];matches=[]
            if refs and pred:
                overlap=np.array([[iou(obj['box'],p['box']) for p in pred] for k,obj in refs])
                left,right=linear_sum_assignment(1-overlap)
                for a,b in zip(left,right):
                    if overlap[a,b]<.5:continue
                    p=pred[b];kp=np.asarray(p['corners'],float)
                    ratio=0.;confidence=0.
                    if kp.shape==(8,3) and np.isfinite(kp).all():
                        area=cv2.contourArea(cv2.convexHull(kp[:,:2].astype(np.float32)))
                        box=p['box'];ratio=area/max(1,(box[2]-box[0])*(box[3]-box[1]));confidence=float(np.median(kp[:,2]))
                    matches.append({'track':refs[a][0],'box_iou':float(overlap[a,b]),'cuboid_hull_to_box_area':ratio,
                        'corner_median_confidence':confidence,'passes_structure_screen':bool(.35<=ratio<=2.5 and confidence>=.25)})
            frames.append({'case':name,'frame':row['frame'],'t':row['t'],'reference_boxes':len(refs),
                           'predictions':len(pred),'matched_reference_boxes':len(matches),
                           'matched_cuboids_passing_structure_screen':sum(m['passes_structure_screen'] for m in matches),'matches':matches})
        total=sum(x['reference_boxes'] for x in frames)
        runs.append({'model':'UrbanOmniDetect-2 L','image_size':rows[0].get('image_size',960),'gpu':rows[0]['gpu'],
                     'runtime':rows[0]['runtime'],'frames':frames,'total_reference_boxes':total,
                     'matched_reference_boxes':sum(x['matched_reference_boxes'] for x in frames),
                     'valid_structure_matches':sum(x['matched_cuboids_passing_structure_screen'] for x in frames),
                     'zero_detection_frames':[(x['case'],x['frame']) for x in frames if x['predictions']==0],
                     'output':str(file.relative_to(ROOT).parent.as_posix()),'inference_and_output_seconds':sum(x['seconds'] for x in rows)})
    report={'reference':'Existing source SAM boxes; not human 3D ground truth',
            'screen':'IoU >= .5 for correspondence; convex hull/2D-box area .35..2.5 and median corner confidence >= .25 for basic structure',
            'not_measured':['3D orientation accuracy','metric position accuracy','trajectory accuracy'],
            'runs':runs}
    path=ROOT/'data/research/vision-only-models/benchmark.json';path.write_text(json.dumps(report,indent=2))
    for r in runs:
        print(r['image_size'],r['matched_reference_boxes'],'/',r['total_reference_boxes'],'source boxes matched;',
              r['valid_structure_matches'],'cuboids pass structural screen; zero-detection frames:',r['zero_detection_frames'])
    return report


if __name__=='__main__':evaluate()
