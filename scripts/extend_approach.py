"""Extend the generated empty scene backwards and publish an approach-road replay.

python scripts/extend_approach.py --case cutin_rearend hero
All neural 3D inference/rendering runs offline on Kaggle with the existing saved GPU runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mercy import config
from mercy.analysis import seal
from mercy.ingest.approach import digest, plan, fuse
from mercy.generation.approach import generate
from scripts.kaggle_gpu import client, SLUGS

STAGE = ROOT/'data'/'kaggle-approach'


def gpu(plans, phase):
    if not plans:
        return
    api = client();owner=api.get_config_value('username')
    fingerprint=hashlib.sha256((phase+json.dumps(plans,sort_keys=True)).encode()).hexdigest()
    batch=fingerprint[:8]
    inputs = STAGE/'inputs'/phase/batch
    inputs.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(inputs/'payload.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('approach-jobs.json',json.dumps(plans))
        for item in plans:
            case=config.case_dir(item['case'])
            if phase=='guide':
                bundle.write(case/'environment'/'world.splat',item['case']+'/base.splat')
            else:
                bundle.write(case/'environment'/'approach'/'extended.png',item['case']+'/extended.png')
    ref=f'{owner}/mercy-approach-inputs-{phase}-{batch}'
    (inputs/'dataset-metadata.json').write_text(json.dumps({'id':ref,'title':'MERCY approach inputs','licenses':[{'name':'other'}]}))
    exists=any(d.ref==ref for d in api.dataset_list(mine=True,search=f'mercy-approach-inputs-{phase}-{batch}'))
    if not exists:
        response=api.dataset_create_new(str(inputs),public=False,quiet=True)
        print('Private input upload:',response,flush=True)
        if getattr(response,'error',None):raise RuntimeError(str(response))
    for _ in range(90):
        try:
            if 'ready' in str(api.dataset_status(ref)).lower():break
        except Exception as error:
            if getattr(getattr(error,'response',None),'status_code',None) not in [403,404]:raise
        time.sleep(2)
    else:raise TimeoutError('Private approach dataset not ready')
    kernel=f'{owner}/mercy-approach-{phase}-{batch}'
    folder=STAGE/phase/'kernel'/batch;folder.mkdir(parents=True,exist_ok=True)
    (folder/'worker.py').write_text(f'PHASE={phase!r}\n'+(ROOT/'kaggle'/'approach_worker.py').read_text())
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':kernel,'title':kernel.split('/')[1],
        'code_file':'worker.py','language':'python','kernel_type':'script','is_private':True,
        'enable_gpu':True,'enable_internet':False,'dataset_sources':[ref],
        'kernel_sources':[f'{owner}/{SLUGS["setup"]}']}))
    response=api.kernels_push(str(folder),acc='NvidiaTeslaT4')
    print(response,flush=True)
    if response.error or response.invalid_dataset_sources or response.invalid_kernel_sources:
        raise RuntimeError('Kaggle rejected the approach job')
    deadline=time.monotonic()+2400
    while time.monotonic()<deadline:
        status=api.kernels_status(kernel);print(phase,status,flush=True)
        value=str(status.status).lower()
        if 'complete' in value:break
        if any(word in value for word in ['error','fail','cancel']):
            api.kernels_output(kernel,path=str(STAGE/phase/'failure'),file_pattern=r'\.log$')
            raise RuntimeError(f'Approach {phase} job failed; inspect {STAGE/phase/"failure"}')
        time.sleep(20)
    else:raise TimeoutError(f'Approach job still pending: {kernel}')
    output=STAGE/phase/'output';output.mkdir(parents=True,exist_ok=True)
    api.kernels_output(kernel,path=str(output))


def run(cases, until='publish'):
    plans=[plan(config.case_dir(name)) for name in cases]
    missing=[]
    for item in plans:
        output=STAGE/'guide'/'output'/'approach'/item['case']
        if not (output/'result.json').exists() or json.loads((output/'result.json').read_text()).get('plan_id')!=item['plan_id']:
            missing.append(item)
    gpu(missing,'guide')
    for item in plans:
        src=STAGE/'guide'/'output'/'approach'/item['case']
        if json.loads((src/'result.json').read_text())['plan_id']!=item['plan_id']:raise ValueError('Guide belongs to another camera plan')
        dst=config.case_dir(item['case'])/'environment'/'approach';dst.mkdir(exist_ok=True)
        for name in ['guide.png','known.png','guide-depth.npz']:shutil.copy2(src/name,dst/name)
        (dst/'plan.json').write_text(json.dumps(item,indent=2))
    if until=='guide':return
    for item in plans:
        image=generate(config.case_dir(item['case']),item)
        item['image_sha256']=digest(image)
    if until=='generate':return
    pending=[]
    for item in plans:
        output=STAGE/'predict'/'output'/'approach'/item['case']/'result.json'
        current=json.loads(output.read_text()) if output.exists() else {}
        if current.get('plan_id')!=item['plan_id'] or current.get('image_sha256')!=item['image_sha256']:pending.append(item)
    gpu(pending,'predict')
    if until=='predict':return
    for item in plans:
        case=config.case_dir(item['case'])
        result=fuse(case,item,STAGE/'predict'/'output'/'approach'/item['case'])
        seal.seal(case)
        print(item['case'],f"{item['distance_m']:.0f}m generated approach road",result['alignment'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',nargs='+',required=True)
    parser.add_argument('--until',choices=['guide','generate','predict','publish'],default='publish')
    args=parser.parse_args();run(args.case,args.until)
