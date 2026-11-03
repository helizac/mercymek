"""Validate finished staged examples, then move them into the live case catalog.

python scripts/publish_cases.py london_rearend vegas_rearend --check-only
python scripts/publish_cases.py london_rearend vegas_rearend
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mercy import config
from mercy.analysis.seal import verify
from mercy.ingest.source_quality import validate
from mercy.ingest.traffic_constraints import body_dimensions, swept_conflicts

REQUIRED=['meta.json','source_provenance.json','scene.json','overlays.json','recon.npz','tracks.json',
          'physics.json','clip.mp4','clip_redacted.mp4','evidence/impact.jpg','slowmo.mp4','slowmo.json',
          'moments.json','environment/world.splat','environment/input.json','environment/empty.review.json',
          'vehicles/profiles.json','plates.json','research.json','verdict.json','diagram.png','statement.json','seal.json']


def check(case):
    missing=[name for name in REQUIRED if not (case/name).is_file() or not (case/name).stat().st_size]
    if missing:
        raise ValueError(f'{case.name}: unfinished artifacts: {", ".join(missing)}')
    meta=json.loads((case/'meta.json').read_text())
    provenance=json.loads((case/'source_provenance.json').read_text())
    source=next((p for p in config.RAW_DIR.rglob(meta['source_clip'])
                 if hashlib.sha256(p.read_bytes()).hexdigest()==provenance['sha256']),None)
    if source is None:
        raise ValueError(f'{case.name}: cannot verify the original source video')
    validate(source,case/'source_provenance.json')
    plates=json.loads((case/'plates.json').read_text())['plates']
    original=[p for p in plates if p.get('source_verified') and p.get('legible') and
              p.get('confidence',0)>=.8 and p.get('text') and '?' not in p['text']]
    if not original:
        raise ValueError(f'{case.name}: no verified readable original-frame plate in the retained clip')
    for plate in original:
        frame=case/plate['original_frame_image']
        if hashlib.sha256(frame.read_bytes()).hexdigest()!=plate['original_frame_sha256']:
            raise ValueError(f'{case.name}: original plate frame changed')
        if not 0<=plate['t']<meta['duration']:
            raise ValueError(f'{case.name}: original plate frame is outside the replay')
    physics=json.loads((case/'physics.json').read_text())
    manifest=json.loads((case/'moments.json').read_text())
    if physics['impact']['kind']!='collision' or len(manifest['moments'])<3:
        raise ValueError(f'{case.name}: incomplete collision replay')
    reference=None
    for moment in manifest['moments']:
        if moment['event_kind']!='collision' or not moment.get('reconstruction'):
            raise ValueError(f'{case.name}: a moment has no reconstructed collision')
        contact=moment.get('contacts',{}).get('event_contact')
        if not contact or not -.0002<=contact['surface_gap_after_m']<=contact['max_surface_gap_m']+.0002:
            raise ValueError(f'{case.name}: colliding vehicle bodies do not meet')
        vehicles=moment['vehicles']
        if any(v.get('visibility')!='persistent-world' for v in vehicles):
            raise ValueError(f'{case.name}: a source vehicle can disappear')
        keys=np.array([v['keys'] for v in vehicles],float)
        absolute={v['key']:np.column_stack([keys[i,:,0]+moment['t_frame'],keys[i,:,1:]]) for i,v in enumerate(vehicles)}
        if reference is not None and (set(reference)!=set(absolute) or
                any(reference[k].shape!=absolute[k].shape or not np.allclose(reference[k],absolute[k],atol=2e-6,rtol=0)
                    for k in reference)):
            raise ValueError(f'{case.name}: actor identity or motion changes between moments')
        reference=absolute
        conflicts=swept_conflicts(keys[:,:,1:3],keys[:,:,3],np.array([body_dimensions(v['model']) for v in vehicles]))
        if conflicts:
            raise ValueError(f'{case.name}: {len(conflicts)} intersecting body intervals')
        for name in [moment['file'],moment.get('base_file',moment['file'])]:
            if not (case/name).is_file():raise ValueError(f'{case.name}: missing scene {name}')
    integrity=verify(case)
    if not integrity.get('intact') or integrity.get('added'):
        raise ValueError(f'{case.name}: evidence seal is not intact: {integrity}')
    return {'case':case.name,'title':meta['title'],'source':provenance['source_url'],
            'resolution':[meta['width'],meta['height']],'original_plates':[p['text'] for p in original],
            'moments':len(manifest['moments']),'actors':len(reference),'continuous_conflicts':0,
            'event_contact':manifest['moments'][0]['contacts']['event_contact'],'seal_root':integrity['root']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases',nargs='+')
    parser.add_argument('--staging',type=Path,default=ROOT/'data/staging/cases')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    sources=[args.staging/name for name in args.cases]
    for source in sources:
        if source.resolve().parent!=args.staging.resolve():raise ValueError('Case must be a direct staging child')
        if (config.CASES_DIR/source.name).exists():raise FileExistsError(f'Live case already exists: {source.name}')
    reports=[check(source) for source in sources]
    if not args.check_only:
        config.CASES_DIR.mkdir(parents=True,exist_ok=True)
        for source in sources:source.rename(config.CASES_DIR/source.name)
    output=ROOT/'data/research/new-scenarios/publication.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'published':not args.check_only,'cases':reports},indent=2))
    print(json.dumps({'published':not args.check_only,'cases':[r['case'] for r in reports]},indent=2))


if __name__=='__main__':main()
