import json

import numpy as np
import pytest

from mercy.ingest.wilddet import decode, qualify, local_prior, load
from mercy.ingest.vehicle_scene import project_box
from mercy.ingest.road_motion import fit_lane


def prediction():
    C = np.array([0., 1.4, 0.])
    R = np.diag([1., -1., 1.])
    K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
    # vis4d +X front rotated into camera +Z; dimensions are WIDTH, LENGTH, HEIGHT.
    box3d = [3., .65, 30., 1.8, 4.6, 1.5, 2 ** -.5, 0., -(2 ** -.5), 0.]
    box = project_box([3., 30.], 0., [1.8, 1.5, 4.6], C, R, K).tolist()
    return dict(track='car:1', source_box=box, box=box, box3d=box3d, score2d=.95, score3d=.8), dict(C=C, R=R, K=K, t=1., frame=30)


def test_vis4d_dimension_order_and_front_axis_transform_to_shared_road():
    raw, job = prediction()
    p = decode(raw['box3d'], job['C'], job['R'])
    assert p['dimensions'] == [1.8, 1.5, 4.6]
    assert p['position'] == pytest.approx([3., 30.])
    assert p['yaw'] == pytest.approx(0., abs=1e-10)
    assert p['ground_y'] == pytest.approx(0., abs=1e-10)
    anchor, reason = qualify(raw, job, np.array([1.8, 1.5, 4.6]))
    assert reason is None and anchor['reprojection_iou'] == pytest.approx(1.)


@pytest.mark.parametrize('mutation,reason', [
    (lambda p: p.update(score3d=.01), 'low_confidence'),
    (lambda p: p.update(box=[0, 0, 20, 20]), 'wrong_2d_association'),
    (lambda p: p['box3d'].__setitem__(3, 12.), 'implausible_dimensions'),
    (lambda p: p['box3d'].__setitem__(0, 12.), '3d_reprojection_mismatch'),
    (lambda p: p['box3d'].__setitem__(6, float('nan')), 'malformed_3d'),
])
def test_bad_neural_predictions_cannot_replace_source_geometry(mutation, reason):
    p, j = prediction(); mutation(p)
    anchor, failure = qualify(p, j)
    assert anchor is None and failure == reason


def test_no_neural_extrapolation_across_missing_or_large_gap_observations():
    anchors = [dict(t=1., position=[0, 10], yaw=179), dict(t=1.5, position=[0, 12], yaw=-179)]
    assert local_prior(anchors, .8) is None
    assert local_prior(anchors, 2.) is None
    assert local_prior(anchors, 1.25)['position'] == pytest.approx([0, 11])
    assert abs(local_prior(anchors, 1.25)['yaw']) == pytest.approx(180)
    assert local_prior([anchors[0], dict(anchors[1], t=5.)], 1.2) is None


def test_stale_calibration_fails_instead_of_silently_using_old_neural_anchors(tmp_path):
    (tmp_path / 'vehicles').mkdir(); (tmp_path / 'overlays.json').write_text('{}')
    (tmp_path / 'vehicles/wilddet3d.json').write_text(json.dumps(dict(overlays_sha256='old')))
    with pytest.raises(ValueError, match='Stale'):
        load(tmp_path)


def test_neural_position_anchors_cannot_bend_or_reverse_normal_traffic():
    from test_road_motion import fixture_scene
    t, c, dims, true, boxes = fixture_scene()
    anchors = [dict(t=float(t[i]), position=(true[i] + [(-1) ** i * 2, (-1) ** i * 3]).tolist()) for i in range(0, len(t), 5)]
    result, info = fit_lane(t, boxes, np.ones_like(boxes, dtype=bool), [c.at(x) for x in t], dims, true, t, neural_anchors=anchors)
    assert np.ptp(result[:, 0]) < 1e-8
    assert np.all(np.diff(result[:, 1]) >= -1e-8)
    assert np.all(result[:, 2] == 0.)
    assert info['neural_anchor_count'] == len(anchors)
    assert np.max(abs(result[:, :2] - true)) < .5
