"""Per-track vehicle appearance from original footage; never use the overlay palette as paint.

Models are complete parametric 3D vehicle bodies with observed type/paint/plate,
not a claim to have recovered unseen bodywork or crash deformation.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

import cv2

from mercy import config
from mercy.meta_client import data_url, structured

TYPES = ["sedan", "hatchback", "rounded_hatchback", "suv", "offroad_suv", "crossover", "wagon", "mpv", "fastback", "coupe", "pickup", "van", "box_truck", "semi_truck", "tanker", "dump_truck", "bus", "motorcycle"]
DIMENSIONS = {"sedan": [1.82, 1.47, 4.7], "hatchback": [1.78, 1.48, 4.15], "suv": [1.92, 1.75, 4.8],
              "coupe": [1.85, 1.35, 4.5], "pickup": [2.0, 1.85, 5.5], "van": [2.0, 2.3, 5.4],
              "box_truck": [2.5, 3.5, 8.0], "semi_truck": [2.55, 3.9, 16.0], "tanker": [2.55, 3.6, 12.0],
              "dump_truck": [2.5, 3.3, 8.0], "bus": [2.55, 3.2, 12.0], "motorcycle": [0.8, 1.2, 2.1]}
DIMENSIONS.update(crossover=[1.86,1.64,4.55], wagon=[1.84,1.50,4.75], mpv=[1.92,1.86,4.90], fastback=[1.86,1.47,4.75])
DIMENSIONS.update(rounded_hatchback=[1.75,1.52,3.9], offroad_suv=[1.98,1.87,4.65])
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["body_type", "paint_hex", "paint_name", "confidence", "notes", "headings"],
          "properties": {"body_type": {"type": "string", "enum": TYPES}, "paint_hex": {"type": "string"},
                         "paint_name": {"type": "string"}, "confidence": {"type": "number"}, "notes": {"type": "string"},
                         "headings": {"type":"array","items":{"type":"object","additionalProperties":False,
                           "required":["source_frame","relative_yaw_deg","confidence","view"],"properties":{
                            "source_frame":{"type":"integer"},"relative_yaw_deg":{"type":"number"},
                            "confidence":{"type":"number"},"view":{"type":"string","enum":["rear","front","left","right","uncertain"]}}}}}}

VEHICLE_CONCEPTS = {"car","truck","bus","van","motorcycle","motorbike","pickup","vehicle"}


def vehicle_observations(overlays):
    """Use source detections, including cars without a usable neural-depth track."""
    result = {}
    for frame, objects in overlays.items():
        for key, obj in objects.items():
            if obj.get('concept','').lower() not in VEHICLE_CONCEPTS:
                continue
            x1,y1,x2,y2=obj['box']
            if x2-x1>=7 and y2-y1>=6:
                result.setdefault(key,[]).append((int(frame),obj))
    return {key:sorted(rows) for key,rows in result.items() if len(rows)>=3}


def plate_for(key: str, plates: dict) -> dict:
    candidates = [p for p in plates.get("plates", []) if p.get("vehicle") == key]
    best = max(candidates, key=lambda p: (p.get("legible", False), p.get("source_verified", False), p.get("confidence", 0)), default=None)
    if plates.get("blurred_at_source") or best is None:
        return {"text": "", "observed": False, "reason": "source blurred" if plates.get("blurred_at_source") else "not readable"}
    # Preserve '?' from OCR, even for partially legible readings. Never guess missing characters.
    return {"text": best.get("text", "") if best.get("legible") else "", "observed": bool(best.get("legible")),
            "confidence": best.get("confidence", 0), "source_key": best["key"], "notes": best.get("notes", "")}


def profiles(case: pathlib.Path, force=False, workers=3) -> dict:
    meta = json.loads((case / "meta.json").read_text())
    overlays = json.loads((case / "overlays.json").read_text())
    plates = json.loads((case / "plates.json").read_text()) if (case / "plates.json").exists() else {}
    dst = case / "vehicles"
    dst.mkdir(exist_ok=True)
    path = dst / "profiles.json"
    cached = json.loads(path.read_text()) if path.exists() and not force else {}
    observations=vehicle_observations(overlays)
    jobs=[]
    # Each crop has its own full-frame context, so a foreground occluder is not mistaken for the tracked car.
    cap=cv2.VideoCapture(str(case/'clip.mp4'))
    try:
        for key,rows in observations.items():
            chosen=[]
            for segment in [rows[:max(1,len(rows)//3)],rows[len(rows)//3:max(len(rows)//3+1,2*len(rows)//3)],rows[2*len(rows)//3:]]:
                def quality(row):
                    x1,y1,x2,y2=row[1]['box'];area=(x2-x1)*(y2-y1)
                    return area*(.3 if x1<2 or x2>meta['width']-3 or y2>meta['height']*.83 else 1)
                if segment:
                    row=max(segment,key=quality)
                    if row[0] not in [x[0] for x in chosen]:chosen.append(row)
            images=[]
            stem=re.sub(r'[^\w-]','_',key)
            for f,obj in chosen:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read()
                if not ok:continue
                x1,y1,x2,y2=map(int,obj['box'])
                crop=image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)]
                if not crop.size:continue
                crop_path=dst/f'{stem}.f{f:04d}.png';cv2.imwrite(str(crop_path),crop)
                context=image.copy();cv2.rectangle(context,(x1,y1),(x2,y2),(0,230,255),3)
                context=cv2.resize(context,(768,round(image.shape[0]*768/image.shape[1])))
                context_path=dst/f'{stem}.f{f:04d}.context.jpg';cv2.imwrite(str(context_path),context)
                images.append((f,crop_path,context_path))
            if not images:continue
            signature=hashlib.sha256(('appearance-visible-heading-v3'+''.join(
                hashlib.sha256(p.read_bytes()).hexdigest() for _,a,b in images for p in [a,b])).encode()).hexdigest()
            jobs.append((key,images,signature))
    finally:cap.release()

    def identify(job):
        key,images,signature=job
        old=cached.get(key,{})
        if old.get('signature')==signature and not force:
            return key,dict(old,plate=plate_for(key,plates))
        content=[{'type':'input_text','text':
            'Describe ONE persistent vehicle for a 3D video replay. Each pair is an ORIGINAL source crop and '
            'the full frame with that track outlined in yellow. Follow the outlined car, not a nearer occluder. '
            'Use all views for body type and paint #RRGGBB. Distinguish a separate sedan trunk from a rounded '
            'hatchback/fastback tailgate, SUV/crossover and tall MPV. Use rounded_hatchback for a fully '
            'domed rear/body (Beetle/500-like), and offroad_suv for an upright Jeep-like cabin and rugged '
            'wheels; ordinary SUV/crossover remains a separate class. Ignore overlay colours/reflections. '
            'Do not read or invent plates. For EACH supplied source_frame estimate the vehicle NOSE heading '
            'on the road relative to camera forward: 0 degrees points AWAY from the camera (taillights/rear '
            'visible), 180 points TOWARDS it (headlights/front visible), +90 points towards IMAGE RIGHT, '
            '-90 towards IMAGE LEFT. This is body orientation, NEVER inferred from screen motion or speed. '
            'A car slowing or stationary still has the same nose direction. Side perspective alone is not '
            'a 90-degree turn. Give conservative angles and confidence 0..1. Explain occlusion/uncertainty.'}]
        for f,crop,context in images:
            content += [{'type':'input_text','text':f'source_frame={f}: crop, then context'},
                        {'type':'input_image','image_url':data_url(crop)},
                        {'type':'input_image','image_url':data_url(context)}]
        appearance=structured(config.VISION_MODEL,[{'role':'user','content':content}],SCHEMA,'vehicle_multiview_appearance',timeout=150)
        if not re.fullmatch(r'#[0-9a-fA-F]{6}',appearance['paint_hex']):raise ValueError(f'Invalid paint for {key}')
        valid_frames={f for f,_,_ in images}
        if not appearance['headings'] or any(h['source_frame'] not in valid_frames or not -360<=h['relative_yaw_deg']<=360 for h in appearance['headings']):
            raise ValueError(f'Invalid source heading samples for {key}')
        f,crop,_=max(images,key=lambda x:x[1].stat().st_size)
        profile=dict(appearance,key=key,dimensions_m=DIMENSIONS[appearance['body_type']],signature=signature,
          crop=crop.relative_to(case).as_posix(),crop_sha256=hashlib.sha256(crop.read_bytes()).hexdigest(),
          source_frame=f,plate=plate_for(key,plates),geometry='rounded parametric complete body; unseen surfaces estimated',
          orientation_method='source-image nose/rear evidence; not the derivative of noisy 3D positions')
        print(case.name,key,profile['body_type'],profile['paint_name'],flush=True)
        return key,profile

    out={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key,profile in pool.map(identify,jobs):
            out[key]=profile
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(path)
    return out
