"""Re-register all observed vehicles, preserving the static world and generated road extension."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mercy import config
from mercy.analysis import seal
from mercy.ingest.vehicle_scene import publish

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',nargs='+',required=True)
    p.add_argument('--motion-only',action='store_true',help='joint lane-motion fit from cached unconstrained source fits')
    args=p.parse_args()
    for name in args.case:
        case=config.case_dir(name)
        if args.motion_only:
            from mercy.ingest.road_motion import regularize_existing
            info=regularize_existing(case);seal.seal(case)
            print(name,len(info['tracks'])-1,'source tracks constrained to road motion',flush=True)
        else:
            info=publish(case);seal.seal(case)
            print(name,info['vehicle_count'],'source vehicles registered',flush=True)
