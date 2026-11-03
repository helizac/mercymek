"""Test an RGB-only vehicle-corner model on Kaggle, persisting its setup for offline inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mercy import config
from scripts.kaggle_gpu import client

STAGE=ROOT/'data/kaggle-vehicle-pose'
SETUP='mercy-rgb-vehicle-runtime-v1'


def http_status(error):
    seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        code=getattr(getattr(error,'response',None),'status_code',None)
        if code:return code
        error=error.__cause__ or error.__context__
    return None


def wait(api,ref,output=None):
    deadline=time.monotonic()+2400
    permission_deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        try:status=api.kernels_status(ref)
        except Exception as e:
            if http_status(e) in [403,404] and time.monotonic()<permission_deadline:
                print('Waiting for new private kernel visibility',ref,flush=True);time.sleep(10);continue
            raise
        print(ref,status,flush=True);state=str(status.status).lower()
        if 'complete' in state:
            if output:api.kernels_output(ref,path=str(output))
            return
        if any(s in state for s in ['error','fail','cancel']):
            dst=STAGE/'failure'/ref.split('/')[-1];dst.mkdir(parents=True,exist_ok=True)
            api.kernels_output(ref,path=str(dst),file_pattern=r'\.log$')
            raise RuntimeError(f'RGB-only detector job failed; inspect {dst}')
        time.sleep(20)
    raise TimeoutError(f'Kaggle job still pending: {ref}')


def push(api,owner,phase,slug,datasets=(),image_size=960):
    folder=STAGE/phase/slug;folder.mkdir(parents=True,exist_ok=True)
    (folder/'worker.py').write_text(f'PHASE={phase!r}\nIMAGE_SIZE={image_size!r}\n'+(ROOT/'kaggle/vehicle_pose_worker.py').read_text())
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':f'{owner}/{slug}','title':slug,
       'code_file':'worker.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,
       'enable_internet':phase=='setup','dataset_sources':list(datasets),
       'kernel_sources':[] if phase=='setup' else [f'{owner}/{SETUP}']}))
    result=api.kernels_push(str(folder),acc='NvidiaTeslaT4');print(result,flush=True)
    if result.error or result.invalid_dataset_sources or result.invalid_kernel_sources:raise RuntimeError('Kaggle rejected the RGB-only experiment')


def run(names,image_size=960):
    api=client();owner=api.get_config_value('username')
    try:state=str(api.kernels_status(f'{owner}/{SETUP}').status).lower()
    except Exception as e:
        if http_status(e) not in [403,404]:raise
        state='missing'
    if state=='missing' or any(s in state for s in ['error','fail','cancel']):push(api,owner,'setup',SETUP)
    wait(api,f'{owner}/{SETUP}')
    inputs=STAGE/'inputs';inputs.mkdir(parents=True,exist_ok=True);jobs=[]
    for name in names:
        c=config.case_dir(name);meta=json.loads((c/'meta.json').read_text());cap=cv2.VideoCapture(str(c/'clip.mp4'))
        try:
            for t in ([.95,4.02,4.56] if name=='cutin_rearend' else [5.5,8.,9.3]):
                f=round(t*meta['src_fps']);cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read()
                if not ok:raise ValueError(f'Cannot read original frame {name}:{f}')
                path=inputs/f'{name}-{f:04d}.png';cv2.imwrite(str(path),image)
                jobs.append({'case':name,'frame':f,'t':f/meta['src_fps'],'image':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        finally:cap.release()
    fingerprint=hashlib.sha256(json.dumps(jobs,sort_keys=True).encode()+str(image_size).encode()+(ROOT/'kaggle/vehicle_pose_worker.py').read_bytes()).hexdigest()[:10]
    ref=f'{owner}/mercy-rgb-pose-inputs-{fingerprint}'
    with zipfile.ZipFile(inputs/'pose-inputs.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('jobs.json',json.dumps(jobs))
        for j in jobs:bundle.write(inputs/j['image'],j['image'])
    upload=STAGE/'upload'/fingerprint;upload.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(inputs/'pose-inputs.zip',upload/'pose-inputs.zip')
    (upload/'dataset-metadata.json').write_text(json.dumps({'id':ref,'title':'MERCY original RGB vehicle pose inputs','licenses':[{'name':'other'}]}))
    for previous in sorted((STAGE/'upload').glob('*/dataset-metadata.json')):
        if previous.parent==upload:continue
        with zipfile.ZipFile(previous.parent/'pose-inputs.zip') as bundle:
            same=json.loads(bundle.read('jobs.json'))==jobs
        if same:
            ref=json.loads(previous.read_text())['id'];upload=previous.parent;break
    exists=any(x.ref==ref for x in api.dataset_list(mine=True,search=ref.split('/')[-1]))
    if not exists:
        result=api.dataset_create_new(str(upload),public=False,quiet=True)
        if getattr(result,'error',None):raise RuntimeError(str(result))
    for _ in range(90):
        try:
            if 'ready' in str(api.dataset_status(ref)).lower():break
        except Exception as e:
            if getattr(getattr(e,'response',None),'status_code',None) not in [403,404]:raise
        time.sleep(2)
    else:raise TimeoutError('Private RGB input dataset did not become ready')
    slug=f'mercy-rgb-pose-test-{fingerprint}';output=STAGE/'output'/fingerprint;output.mkdir(parents=True,exist_ok=True)
    if not (output/'pose-predictions/results.json').exists():
        push(api,owner,'infer',slug,[ref],image_size);wait(api,f'{owner}/{slug}',output)
    (STAGE/'latest.json').write_text(json.dumps({'kernel':f'{owner}/{slug}','output':str(output.relative_to(ROOT)),'source_frames':len(jobs)},indent=2))
    print('RGB-only experiment saved:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',nargs='+',required=True)
    p.add_argument('--imgsz',type=int,default=960,choices=[640,960]);args=p.parse_args();run(args.case,args.imgsz)
