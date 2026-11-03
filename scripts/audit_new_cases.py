"""Native-frame projection fixtures for independent browser checks of new cases."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mercy.ingest.playback_motion import camera_path
from mercy.ingest.vehicle_scene import Cameras,continuous_cameras


def fixtures(case):
    meta=json.loads((case/'meta.json').read_text());m=json.loads((case/'moments.json').read_text())['moments'][0]
    plate=next(p for p in json.loads((case/'plates.json').read_text())['plates'] if p.get('source_verified'))
    with np.load(case/'recon.npz') as rec:
        times=rec['times'];K=rec['intrinsic'].copy();height,width=rec['depth'].shape[1:]
    K*=np.array([[meta['width']/width],[meta['height']/height],[1.]])
    with np.load(case/'environment/cameras.npz') as rec:matrices=rec['matrices']
    matrices=continuous_cameras(times,matrices,m['replay_start'],m['replay_end'])
    anchor=json.loads((case/'environment/input.json').read_text())['anchor_t']
    matrices,velocity=camera_path(case,times,matrices,anchor,return_velocity=True,write_report=False)
    cameras=Cameras(times,matrices,K,position_velocity=velocity)
    result=[]
    for target in [plate['t'],m['impact_t']-.3,m['impact_t'],min(m['impact_t']+.8,m['replay_end']-.1)]:
        t=np.floor(target*meta['src_fps']+1e-6)/meta['src_fps']
        C,R,lens=cameras.at(t);points=np.array([[.4,-.2,8],[-1.,.3,12],[.2,.8,4.]])
        pixels=points[:,:2]/points[:,2,None]*[lens[0,0],lens[1,1]]+[lens[0,2],lens[1,2]]
        result.append({'time':float(t),'points_road':(points@R.T+C).tolist(),'pixels':pixels.tolist()})
    out=ROOT/'data/research/new-scenarios'/f'{case.name}-projection.json';out.write_text(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir',type=Path,default=ROOT/'data/cases')
    parser.add_argument('cases',nargs='+');args=parser.parse_args()
    for name in args.cases:fixtures(args.cases_dir/name)
