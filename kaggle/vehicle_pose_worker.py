"""Private RGB-only detector experiment. PHASE is prepended; setup output is reused offline."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

SOURCE_COMMIT='caa8537803f61b5d04602a25432575f36fce085d'
WEIGHTS_REV='be4f69a300c43f6d2c7dd35f99b6c2a828912817'
IMAGE_SIZE=globals().get('IMAGE_SIZE',960)
WORK=Path('/kaggle/working');INPUT=Path('/kaggle/input')
os.environ['YOLO_CONFIG_DIR']=str(WORK/'ultralytics-settings')
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
os.environ['YOLO_AUTOINSTALL']='False'


def setup():
    cache=WORK/'mercy-rgb-pose-runtime';cache.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(cache/'site'),
                    'ultralytics==8.4.61','ultralytics-thop==2.0.18'],check=True)
    source=cache/'source';(source/'v2').mkdir(parents=True)
    for name in ['v2/__init__.py','v2/hybrid_v2.py']:
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/mkturkcan/urbanomnidetect/{SOURCE_COMMIT}/{name}',source/name)
    checkpoint=cache/'urbanomnidetect_v2_l_640.pt'
    urllib.request.urlretrieve(f'https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/{WEIGHTS_REV}/checkpoints/urbanomnidetect_v2_l_640.pt',checkpoint)
    sys.path[:0]=[str(source),str(cache/'site')]
    import torch
    import numpy as np
    import v2.hybrid_v2
    from ultralytics import YOLO
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the detector experiment')
    model=YOLO(str(checkpoint))
    model.predict(np.zeros((720,1280,3),np.uint8),imgsz=960,conf=.1,device=0,verbose=False,save=False)
    manifest={'source_commit':SOURCE_COMMIT,'weights_revision':WEIGHTS_REV,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'checkpoint':checkpoint.name,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(0),'ultralytics':'8.4.61','radar_required':False,'depth_sensor_required':False}
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('RGB-only pose runtime saved; GPU smoke test passed',flush=True)


def infer():
    import torch
    import cv2
    import numpy as np
    cache=next(INPUT.rglob('mercy-rgb-pose-runtime/manifest.json')).parent
    runtime=json.loads((cache/'manifest.json').read_text())
    if runtime['torch']!=torch.__version__ or runtime['python'].split()[0]!=sys.version.split()[0]:raise RuntimeError('Saved runtime needs rebuilding for this Kaggle image')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; refusing local/CPU inference fallback')
    sys.path[:0]=[str(cache/'source'),str(cache/'site')]
    import v2.hybrid_v2
    from ultralytics import YOLO
    src=WORK/'inputs'
    archive=next(INPUT.rglob('pose-inputs.zip'),None)
    if archive:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not (src/member.filename).resolve().is_relative_to(src.resolve()):raise ValueError('Unsafe payload path')
            bundle.extractall(src)
    else:
        # Kaggle may automatically extract uploaded ZIP datasets before mounting them.
        src=next(INPUT.rglob('jobs.json')).parent
    model=YOLO(str(cache/runtime['checkpoint']))
    jobs=json.loads((src/'jobs.json').read_text());out=WORK/'pose-predictions';out.mkdir()
    all_results=[]
    for job in jobs:
        image=src/job['image']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=job['sha256']:raise ValueError('Source frame hash mismatch')
        start=time.time()
        with torch.inference_mode():
            r=model.predict(str(image),imgsz=IMAGE_SIZE,conf=.1,device=0,verbose=False,save=False,classes=[1,2,3,5,7])[0]
        predictions=[]
        boxes=r.boxes.xyxy.cpu().numpy();scores=r.boxes.conf.cpu().numpy();classes=r.boxes.cls.cpu().numpy()
        keypoints=r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.empty((len(boxes),0,3))
        visual=cv2.imread(str(image))
        for box,score,cls,kp in zip(boxes,scores,classes,keypoints):
            predictions.append({'box':box.tolist(),'confidence':float(score),'coco_class':int(cls),'corners':kp.tolist()})
            x1,y1,x2,y2=map(int,box);cv2.rectangle(visual,(x1,y1),(x2,y2),(220,180,20),1)
            if len(kp)==8:
                for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                    if min(kp[a,2],kp[b,2])>.25:
                        A,B=tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int));cv2.line(visual,A,B,(30,230,100) if a<4 and b<4 else (40,150,255),1,cv2.LINE_AA)
        cv2.putText(visual,'UrbanOmniDetect-2 RGB-only predictions',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,230,230),2,cv2.LINE_AA)
        name=f'{job["case"]}-{job["frame"]:04d}'
        cv2.imwrite(str(out/(name+'.jpg')),visual)
        result=dict(job,predictions=predictions,seconds=time.time()-start,gpu=torch.cuda.get_device_name(0),runtime=runtime,image_size=IMAGE_SIZE)
        (out/(name+'.json')).write_text(json.dumps(result,indent=2));all_results.append(result)
        print('DONE',name,len(predictions),'RGB-only detections',flush=True)
    (out/'results.json').write_text(json.dumps(all_results,indent=2))


if PHASE=='setup':setup()
elif PHASE=='infer':infer()
else:raise ValueError(PHASE)
