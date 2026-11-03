"""Audit published neural anchors, persistent identities, motion invariants and unchanged static worlds."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config
from mercy.analysis.seal import verify
from mercy.ingest.wilddet import load, file_hash


def check(names):
    cases = {}
    for name in names:
        case = config.case_dir(name)
        neural = load(case)
        assert neural, f'{name}: no imported neural results'
        manifest = json.loads((case / 'moments.json').read_text())
        before_dir = ROOT / 'data/research/wilddet3d/before' / name
        before = json.loads((before_dir / 'moments.json').read_text())
        static = json.loads((before_dir / 'static-hashes.json').read_text())
        for filename, digest in static.items():
            assert file_hash(case / filename) == digest, f'{name}: static/source artifact changed: {filename}'
        normal = 0; supported = 0; dimensions = 0; changes = {}
        first = manifest['moments'][0]
        previous = {v['key']: v for v in before['moments'][0]['vehicles']}
        assert set(previous) == {v['key'] for v in first['vehicles']}, f'{name}: identities changed'
        for m in manifest['moments']:
            assert m['reconstruction'] == 'wilddet-rgb-road-vehicles-v4'
            assert m['wilddet3d']['accepted'] == len(neural['accepted'])
        for v in first['vehicles']:
            old = previous[v['key']]
            assert v.get('observed_range') == old.get('visible_range'), f'{name}: source observation range changed for {v["key"]}'
            assert v.get('visibility') == 'persistent-world' and 'visible_range' not in v, f'{name}: detection still controls existence'
            keys = np.asarray(v['keys'], float)
            assert np.isfinite(keys).all() and (np.diff(keys[:, 0]) > 0).all()
            if v.get('road_motion', {}).get('straight_prior_applied'):
                angle = np.radians(v['road_motion']['heading_deg'])
                forward = np.array([np.sin(angle), np.cos(angle)])
                right = np.array([np.cos(angle), -np.sin(angle)])
                assert np.ptp(keys[:, 1:3] @ right) < 1e-4, f'{name}: sideways drift: {v["key"]}'
                assert np.min(np.diff(keys[:, 1:3] @ forward)) > -1e-4, f'{name}: reversed: {v["key"]}'
                assert np.ptp(keys[:, 3]) < 1e-4, f'{name}: body heading drift: {v["key"]}'
                normal += 1
            registration = v.get('registration', {})
            supported += registration.get('neural_supported_observations', 0) > 0
            dimensions += 'WildDet3D' in v['model'].get('dimensions_source', '')
            old_keys = np.asarray(old['keys'])
            old_xz = np.column_stack([np.interp(keys[:, 0], old_keys[:, 0], old_keys[:, i]) for i in [1, 2]])
            changes[v['key']] = dict(neural_anchors=registration.get('neural_anchor_count', 0),
                supported_observations=registration.get('neural_supported_observations', 0),
                median_position_change_m=float(np.median(np.linalg.norm(keys[:, 1:3] - old_xz, axis=1))),
                old_box_error_px=old.get('registration', {}).get('median_box_error_px'),
                new_box_error_px=registration.get('median_box_error_px'),
                method=registration.get('neural_backend', 'camera-vehicle prior'))
        integrity = verify(case)
        assert integrity['intact'] and not integrity.get('added'), f'{name}: seal mismatch: {integrity}'
        cases[name] = dict(source_frames=neural['frames'], neural_predictions=neural['predictions'],
            qualified_anchors=len(neural['accepted']), rejected_anchors=len(neural['rejected']),
            rejection_reasons=dict(Counter(r['reason'] for r in neural['rejected'])),
            supported_vehicle_tracks=supported, dimension_estimated_tracks=dimensions,
            straight_tracks_checked=normal, vehicle_tracks=len(first['vehicles']) - 1, moments=len(manifest['moments']),
            static_hashes_unchanged=True, identities_and_observation_intervals_unchanged=True,
            physical_actor_lifetime='persistent across whole replay; observations are not existence',
            motion_version=first.get('motion_version'),integrity=intact_summary(integrity), tracks=changes)
        print(name, supported, 'supported tracks;', len(neural['accepted']), '/', neural['predictions'], 'anchors;', normal, 'straight tracks checked', flush=True)
    latest = json.loads((ROOT / 'data/kaggle-wilddet/latest.json').read_text())
    raw = ROOT / latest['output'].replace('\\', '/') / 'wilddet-predictions'
    parity = json.loads((raw / 'batch-validation.json').read_text())
    assert parity['independent_comparison'] and parity['max_box3d_parameter_error'] < .005
    report = dict(at=datetime.now(timezone.utc).isoformat(), backend='WildDet3D RGB', cases=cases,
        gpu_execution=latest, batch_parity=parity, runtime=neural['runtime'],
        limitation='Monocular estimates and consistency gates; no labelled 3D ground truth or certified metric accuracy.')
    (ROOT / 'data/wilddet-run.json').write_text(json.dumps(report, indent=2))
    return report


def intact_summary(result):
    return dict(intact=result['intact'], added=result.get('added', []))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', nargs='+', required=True)
    check(p.parse_args().case)
