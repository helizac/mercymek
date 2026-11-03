"""Muse road continuation anchored to an actual posed 3D render and a protected known-pixel mask."""
import base64
import hashlib
import json
from contextlib import ExitStack
from openai.types import ImagesResponse

import cv2
import numpy as np

from mercy import config
from mercy.meta_client import data_url, meta_client, structured

SCHEMA = {'type':'object','additionalProperties':False,
 'required':['vehicle_free','road_continuous','lighting_matches','known_geometry_preserved','road_polygon','feedback'],
 'properties':{**{key:{'type':'boolean'} for key in ['vehicle_free','road_continuous','lighting_matches','known_geometry_preserved']},
  'road_polygon':{'type':'array','items':{'type':'array','items':{'type':'integer'}}},'feedback':{'type':'string'}}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(case, plan, attempts=3):
    root = case/'environment'/'approach'
    target, review_path = root/'extended.png', root/'review.json'
    guide, original = root/'guide.png', case/'environment'/'empty.png'
    signature = hashlib.sha256((digest(guide)+digest(original)+plan['plan_id']+'feathered-distant-site-v4').encode()).hexdigest()
    if target.exists() and review_path.exists():
        review = json.loads(review_path.read_text())
        if review.get('accepted') and review.get('signature')==signature and review.get('image_sha256')==digest(target):
            return target
    mask = cv2.imread(str(root/'known.png'),cv2.IMREAD_GRAYSCALE)
    mask = cv2.erode((mask>0).astype(np.uint8),np.ones((5,5),np.uint8))>0
    # High opacity is not a guarantee of accurate unseen geometry. Protect distant central landmarks;
    # let Muse repair stretched peripheral splats exposed by the much larger camera translation.
    mask[:,:round(plan['width']*.46)]=False;mask[:,round(plan['width']*.54):]=False
    mask[:round(plan['height']*.47)]=False;mask[round(plan['height']*.64):]=False
    weight = np.clip(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)/28,0,1)
    cv2.imwrite(str(root/'protected.png'),(weight>.98).astype(np.uint8)*255)
    rendered = cv2.imread(str(guide))
    scene = json.loads((case/'scene.json').read_text())
    weather = scene.get('weather','unknown')
    road_type = scene.get('road_type','road')
    structure = ('The SAME elevated concrete road deck MUST continue directly OVERHEAD from the distant site '
                 'all the way above the new camera position, with matching left-side support pillars. '
                 if 'under elevated' in road_type.lower() else '')
    surface = ('DRY matte asphalt: absolutely no puddles, wet gloss or mirror reflections.' if weather=='dry' else
               'Snow-covered road and overcast winter light; no freshly exposed dry asphalt.' if 'snow' in weather.lower() else
               'Match the original road wetness and weather exactly.')
    prompt = (
        'IMAGE 1 is a geometric 3D render GUIDE from a camera translated BACKWARDS along the SAME road, '
        f"{plan['distance_m']:.0f} metres before IMAGE 2, at the SAME camera height {plan['camera_height_m']:.2f}m and SAME heading. "
        'IMAGE 2 is the original EMPTY scene, provided as the appearance reference. '+surface+' '+structure+' Produce ONE photorealistic image '
        'from the exact camera pose of IMAGE 1. Extend the road towards the viewer so a car can drive from this '
        'approach into the distant original scene. Preserve the small central distant site, horizon and lane '
        'perspective, light, weather and location style. Flat untextured gray areas and black gaps in IMAGE 1 '
        'are missing geometry to complete photorealistically, not objects. Distorted peripheral fragments and '
        'stretched blobs are reconstruction artifacts: replace those freely with coherent architecture. '
        'The foreground lane guide is approximate: '
        'continue the source road naturally, with matching lane widths, barriers/curbs and roadside context. '
        'For snow keep snow-covered lanes; for rain keep wet reflections. No vehicles, pedestrians, vehicle shadows, '
        'dashboard, hood, text overlays or added logos. Do not return the original viewpoint or a collage. '
        'The original crash site must remain in the DISTANCE as positioned in IMAGE 1.')
    report = {'signature':signature,'plan_id':plan['plan_id'],'accepted':False,'attempts':[],
              'method':'Muse extension of a fixed-pose 3D render; original high-opacity pixels protected',
              'source_sha256':digest(original),'guide_sha256':digest(guide)}
    previous = guide
    for i in range(attempts):
        with ExitStack() as stack:
            files = [stack.enter_context(p.open('rb')) for p in [previous,original]]
            # Muse requires indexed multipart keys; the SDK's image=[...] encodes unsupported image[] keys.
            response = meta_client().with_options(timeout=300).post('/images/edits',cast_to=ImagesResponse,
                body={'model':config.IMAGE_MODEL,'prompt':prompt,'n':1,'size':'1280x720',
                      'reasoning_strength':'high','output_format':'png',
                      'tool_enablement':{'enable_web_search':False,'enable_image_search':False,'enable_shell':False}},
                files=[(f'image[{j}]',image) for j,image in enumerate(files)],
                options={'headers':{'Content-Type':'multipart/form-data'}})
        raw = root/f'extension.attempt-{i+1}.raw.png'
        raw.write_bytes(base64.b64decode(response.data[0].b64_json))
        candidate = cv2.imread(str(raw))
        if candidate is None:
            raise RuntimeError('Muse did not return a valid extension image')
        candidate = cv2.resize(candidate,(plan['width'],plan['height']),interpolation=cv2.INTER_AREA)
        # The project explicitly uses Muse: this geometric composite protects existing world pixels from drift.
        candidate = np.clip(candidate*(1-weight[...,None])+rendered*weight[...,None],0,255).astype(np.uint8)
        path = root/f'extension.attempt-{i+1}.png'
        cv2.imwrite(str(path),candidate)
        result = structured(config.VISION_MODEL,[{'role':'user','content':[
          {'type':'input_text','text':
            'Image 1: sparse 3D guide from a camera moved backwards along the road. Image 2: original forward '
            'empty scene (appearance reference). Image 3: completed approach image from the GUIDE viewpoint. '
            'Strictly check image 3: no vehicles/fragments/shadows; continuous physically plausible road and '
            'lane perspective leading into the distant original site; matching weather and lighting; existing '
        'distant central crash-site landmarks preserved without obvious seams or doubled buildings. Peripheral '
        'distorted splats in the sparse guide may be repaired freely; only the small central distant site is protected. New unseen roadside '
            'content is allowed. Do not require it to match the ORIGINAL camera position. Give concrete '
            'repair feedback. Also trace visible drivable road in IMAGE 3 as one clockwise polygon of 4..24 '
            '[x,y] vertices normalized to 0..1000 on each axis, excluding sidewalks, barriers, buildings and sky.'},
          {'type':'input_image','image_url':data_url(guide)},
          {'type':'input_image','image_url':data_url(original)},
          {'type':'input_image','image_url':data_url(path)}]}],SCHEMA,'approach_review',timeout=120)
        polygon = np.asarray(result['road_polygon'])
        valid_polygon = polygon.ndim==2 and polygon.shape[1]==2 and 4<=len(polygon)<=24 and (polygon>=0).all() and (polygon<=1000).all()
        accepted = valid_polygon and all(result[key] for key in ['vehicle_free','road_continuous','lighting_matches','known_geometry_preserved'])
        report['attempts'].append(dict(result,file=path.name,accepted=bool(accepted)))
        if accepted:
            target.write_bytes(path.read_bytes())
            report.update(accepted=True,image_sha256=digest(target),road_polygon=result['road_polygon'])
        review_path.write_text(json.dumps(report,indent=2))
        print(case.name,'approach review',i+1,result['feedback'],flush=True)
        if accepted:
            return target
        previous = path
        prompt = ('Correct this road continuation, preserving its camera position, height, heading and all existing '
                  'rendered geometry. '+result['feedback']+' Keep it completely vehicle-free; one photorealistic image.')
    raise RuntimeError(f'{case.name}: road extension did not pass visual review; see {review_path}')
