"""Record continuous body checks, real-browser frame/projection checks, and sealed artifact integrity."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mercy.analysis.seal import verify
from audit_contacts import audit


def check():
    names=['cutin_rearend','hero','lanechange','night_rain','snow_pileup']
    directory=ROOT/'data/research/contacts'
    sync=json.loads((directory/'browser-sync.json').read_text())
    assert {v['case'] for v in sync}==set(names)
    navigation=json.loads((ROOT/'data/research/registration-fix/browser-check.json').read_text())
    lifetimes={v['case']:v for v in navigation if 'body_intersections' in v}
    assert set(lifetimes)==set(names)
    reports=[]
    for name in names:
        case=ROOT/'data/cases'/name
        manifest=json.loads((case/'moments.json').read_text());first=manifest['moments'][0]
        before=json.loads((directory/'before'/name/'moments.json').read_text())['moments'][0]
        assert {v['key'] for v in first['vehicles']}=={v['key'] for v in before['vehicles']}
        reference={v['key']:v for v in first['vehicles']}
        for m in manifest['moments']:
            assert m['contacts']['continuous_nonpenetration'] and m['camera_intrinsics'] and m['source_size']
            for v in m['vehicles']:
                assert v['visibility']=='persistent-world' and v['contacts']['continuous_nonpenetration']
                a=reference[v['key']]['keys'];b=v['keys']
                assert len(a)==len(b)
                assert all(abs((x[0]+first['t_frame'])-(y[0]+m['t_frame']))<2e-6 and x[1:]==y[1:] for x,y in zip(a,b))
        report=audit(name)
        assert report['continuous_conflicting_intervals']==0
        event=manifest.get('contacts',{}).get('event_contact')
        if first.get('event_kind')=='collision':
            assert event and -.0002<=event['surface_gap_after_m']<=event['max_surface_gap_m']+.0002
        else:
            assert event is None
        report['contact_version']=manifest['contacts']['version']
        report['event_contact']=event
        life=lifetimes[name]
        assert life['body_intersections']==0 and life['persistent_actor_visibility'] and life['no_despawns'] and life['stable_actor_identities'] and life['shared_meshes_across_moments']
        frame=next(v for v in sync if v['case']==name)
        for sample in frame['seek_samples']:
            assert sample['scene_time']==sample['presented_time'] and sample['world_projection_error_px']<.01
            assert sample['native_video_in_comparison_window'] and sample['decoded_video_different_channels']==0
        assert frame['playing']['clock']==frame['playing']['scene']
        hashes=json.loads((ROOT/'data/research/wilddet3d/before'/name/'static-hashes.json').read_text())
        for filename,digest in hashes.items():assert hashlib.sha256((case/filename).read_bytes()).hexdigest()==digest
        integrity=verify(case)
        assert integrity['intact'] and not integrity['added'],integrity
        report.update(navigation=life,frame_sync=frame,source_and_static_hashes_unchanged=True,seal_intact=True)
        reports.append(report)
        print(name,report['before_conflicting_intervals'],'-> 0 swept conflicts; pixels/clock/projection PASS; seal intact',flush=True)
    logs=[p for p in [directory/'python-tests.log',ROOT/'data/research/impact-alignment/python-tests.log',ROOT/'data/research/ui-vehicles/python-tests.log'] if p.exists()]
    test_log=max(logs,key=lambda p:p.stat().st_mtime)
    tests=test_log.read_text()
    passed=re.search(r'(\d+) passed',tests);assert passed and not re.search(r'\d+ failed',tests)
    result={'at':datetime.now(timezone.utc).isoformat(),'contact_versions':sorted({r['contact_version'] for r in reports}),
        'vehicle_actors_without_ego':sum(v['actors']-1 for v in reports),'case_moments':15,
        'python_tests_passed':int(passed[1]),'before_conflicting_intervals':sum(v['before_conflicting_intervals'] for v in reports),
        'python_test_log':test_log.relative_to(ROOT).as_posix(),
        'final_conflicting_intervals':0,'cases':reports,
        'limitations':['Monocular vehicle placement and class geometry remain estimated; source reprojection errors are retained.',
                      'Nonpenetration constrains kinematic playback; it does not recover deformation or validated collision forces.']}
    (ROOT/'data/contacts-sync-run.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':check()
