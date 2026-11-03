"""Jointly fit source-supported vehicle paths with continuously verified body nonpenetration."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mercy import config
from mercy.ingest.traffic_constraints import publish
from mercy.analysis.seal import seal

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',nargs='+',required=True);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    dst=ROOT/'data/research/contacts';dst.mkdir(parents=True,exist_ok=True)
    for name in args.case:
        report,templates=publish(config.case_dir(name),args.dry_run)
        (dst/(name+'-candidate.json')).write_text(json.dumps({'report':report,'templates':templates},indent=1))
        if not args.dry_run:seal(config.case_dir(name))
        print(name,report,flush=True)
