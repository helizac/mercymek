"""Source-associated, quality-gated RGB WildDet3D anchors in the shared road frame.

WildDet uses vis4d OPENCV cuboids: (cx,cy,cz,width,LENGTH,HEIGHT,qw,qx,qy,qz),
with the length/front axis along local +X. The renderer uses (width,height,length)
and heading zero along road +Z. Never treat the model's quaternion as a browser quaternion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_hash(case):
    with np.load(Path(case) / 'recon.npz') as rec:
        value = rec['intrinsic'].tobytes() + rec['times'].tobytes()
    meta = json.loads((Path(case) / 'meta.json').read_text())
    return hashlib.sha256(value + json.dumps({k: meta[k] for k in ['width', 'height', 'src_fps']}, sort_keys=True).encode()).hexdigest()


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(1e-8, np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - inter))


def decode(box, C, R):
    box = np.asarray(box, float)
    if box.shape != (10,) or not np.isfinite(box).all():
        raise ValueError('nonfinite or malformed 3D box')
    q = box[6:]
    norm = np.linalg.norm(q)
    if not .8 <= norm <= 1.2:
        raise ValueError('invalid quaternion')
    rot = Rotation.from_quat(q[[1, 2, 3, 0]] / norm).as_matrix()
    w, length, height = box[3:6]
    corners = np.array([[x, y, z] for x in [-length / 2, length / 2]
                        for y in [-height / 2, height / 2] for z in [-w / 2, w / 2]])
    camera_corners = corners @ rot.T + box[:3]
    road_corners = camera_corners @ np.asarray(R).T + C
    centre = np.asarray(R) @ box[:3] + C
    front = np.asarray(R) @ rot[:, 0]
    up = np.asarray(R) @ rot[:, 1]
    return dict(position=centre[[0, 2]].tolist(), dimensions=[w, height, length],
        yaw=float(np.degrees(np.arctan2(front[0], front[2]))),
        ground_y=float(road_corners[:, 1].min()), upright=float(abs(up[1])),
        camera_corners=camera_corners)


def qualify(prediction, job, expected_dimensions=None):
    """Reject wrong prompt associations, implausible boxes and incompatible ground/scale estimates."""
    try:
        p = decode(prediction['box3d'], job['C'], job['R'])
    except (ValueError, KeyError, TypeError):
        return None, 'malformed_3d'
    scores = np.array([prediction.get('score2d', 0), prediction.get('score3d', 0)], float)
    if not np.isfinite(scores).all() or scores[0] < .3 or scores[1] < .15:
        return None, 'low_confidence'
    dims = np.array(p['dimensions'])
    if not (.4 <= dims[0] <= 4.5 and .5 <= dims[1] <= 5.5 and 1 <= dims[2] <= 25):
        return None, 'implausible_dimensions'
    if expected_dimensions is not None and ((dims / expected_dimensions < .5).any() or (dims / expected_dimensions > 1.85).any()):
        return None, 'incompatible_vehicle_class'
    if iou(prediction['box'], prediction['source_box']) < .5:
        return None, 'wrong_2d_association'
    corners = p.pop('camera_corners')
    if corners[:, 2].min() <= .25 or corners[:, 2].max() > 300:
        return None, 'invalid_depth'
    K = np.asarray(job['K'])
    pixels = corners @ K.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    projected = np.r_[pixels.min(0), pixels.max(0)]
    overlap = iou(projected, prediction['source_box'])
    if overlap < .35:
        return None, '3d_reprojection_mismatch'
    if p['upright'] < np.cos(np.radians(25)):
        return None, 'tilted_road_vehicle'
    if abs(p['ground_y']) > max(.8, dims[1] * .55):
        return None, 'incompatible_road_plane'
    p.update(frame=int(job['frame']), t=float(job['t']), track=prediction['track'],
        score2d=float(scores[0]), score3d=float(scores[1]), reprojection_iou=overlap,
        source_box=prediction['source_box'], raw_box3d=prediction['box3d'])
    return p, None


def import_results(case, results):
    """Persist only results bound to the exact current source masks, camera poses and image frames."""
    from mercy.ingest.vehicle_scene import Cameras, continuous_cameras
    import cv2
    case = Path(case)
    meta = json.loads((case / 'meta.json').read_text())
    profiles = json.loads((case / 'vehicles/profiles.json').read_text())
    # Profile cache has a provenance envelope; obtain the actual per-track map.
    profiles = profiles.get('profiles', profiles)
    overlays = json.loads((case / 'overlays.json').read_text())
    first = json.loads((case / 'moments.json').read_text())['moments'][0]
    with np.load(case / 'recon.npz') as rec:
        times = rec['times']; K = rec['intrinsic'].copy(); rh, rw = rec['depth'].shape[1:]
    K *= np.array([[meta['width'] / rw], [meta['height'] / rh], [1.]])
    with np.load(case / 'environment/cameras.npz') as cache:
        matrices = cache['matrices']
    cameras = Cameras(times, continuous_cameras(times, matrices, first['replay_start'], first['replay_end']), K)
    rows = [r for r in results if r['case'] == case.name]
    accepted, rejected = [], []
    cap = cv2.VideoCapture(str(case / 'clip.mp4'))
    try:
        for job in rows:
            if job['overlays_sha256'] != file_hash(case / 'overlays.json'):
                raise ValueError('WildDet source masks changed; rerun inference')
            C, R, K = cameras.at(job['t'])
            if not all(np.allclose(job[k], v, rtol=1e-6, atol=1e-6) for k, v in [('C', C), ('R', R), ('K', K)]):
                raise ValueError('WildDet camera calibration changed; rerun inference')
            cap.set(cv2.CAP_PROP_POS_FRAMES, job['frame']); ok, image = cap.read()
            if not ok or hashlib.sha256(cv2.imencode('.png', image)[1].tobytes()).hexdigest() != job['sha256']:
                raise ValueError('WildDet original frame changed')
            for prediction in job['predictions']:
                obj = overlays[str(job['frame'])].get(prediction['track'])
                if obj is None or not np.allclose(obj['box'], prediction['source_box']):
                    raise ValueError('WildDet prompt association no longer matches source')
                dims = profiles.get(prediction['track'], {}).get('dimensions_m')
                anchor, reason = qualify(prediction, job, np.array(dims) if dims else None)
                if anchor:
                    accepted.append(anchor)
                else:
                    rejected.append(dict(frame=job['frame'], track=prediction['track'], reason=reason))
    finally:
        cap.release()
    report = dict(version='wilddet-rgb-v1', backend='WildDet3D', sensors=['RGB'],
        overlays_sha256=file_hash(case / 'overlays.json'), cameras_sha256=file_hash(case / 'environment/cameras.npz'),
        intrinsics_sha256=calibration_hash(case), clip_sha256=file_hash(case / 'clip.mp4'),
        frames=len(rows), predictions=len(accepted) + len(rejected), accepted=accepted, rejected=rejected,
        runtime=rows[0]['runtime'] if rows else None,
        limitations=['monocular scale and orientation remain estimates',
                    'ground-plane/reprojection checks are consistency gates, not 3D ground truth',
                    'unsupported vehicles retain source-box geometric fits; normal traffic retains straight-lane constraints'])
    path = case / 'vehicles/wilddet3d.json'
    path.write_text(json.dumps(report, indent=2))
    return report


def load(case):
    path = Path(case) / 'vehicles/wilddet3d.json'
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    if (report['overlays_sha256'] != file_hash(Path(case) / 'overlays.json')
        or report['cameras_sha256'] != file_hash(Path(case) / 'environment/cameras.npz')
        or report['intrinsics_sha256'] != calibration_hash(case)
        or report['clip_sha256'] != file_hash(Path(case) / 'clip.mp4')):
        raise ValueError('Stale WildDet anchors; source masks/cameras changed')
    return report


def archive_stale(case):
    """A full world rebuild may recalibrate cameras. Preserve old predictions as history, then infer anew."""
    case = Path(case); path = case / 'vehicles/wilddet3d.json'
    if not path.exists():
        return
    try:
        load(case)
    except ValueError:
        history = case / 'vehicles/wilddet3d-history'
        history.mkdir(exist_ok=True)
        path.replace(history / (file_hash(path) + '.json'))


def anchors_for(report, members):
    by_frame = {}
    for anchor in report.get('accepted', []):
        if anchor['track'] not in members:
            continue
        old = by_frame.get(anchor['frame'])
        if old is None or anchor['score3d'] * anchor['reprojection_iou'] > old['score3d'] * old['reprojection_iou']:
            by_frame[anchor['frame']] = anchor
    return sorted(by_frame.values(), key=lambda a: a['t'])


def rebase(report, cameras):
    """Reuse cached camera-frame neural boxes when playback camera translation is regularized."""
    if not report:
        return report
    accepted = []
    for anchor in report['accepted']:
        C, R, _ = cameras.at(anchor['t'])
        converted = decode(anchor['raw_box3d'], C, R)
        accepted.append(dict(anchor, position=converted['position'], yaw=converted['yaw']))
    return dict(report, accepted=accepted)


def local_prior(anchors, t):
    """Interpolate neural anchors only between nearby observations; never extrapolate missing evidence."""
    if not anchors:
        return None
    times = np.array([a['t'] for a in anchors])
    nearest = int(np.argmin(abs(times - t)))
    if abs(times[nearest] - t) > .3:
        return None
    if t < times[0] or t > times[-1]:
        return anchors[nearest] if abs(times[nearest] - t) < .035 else None
    i = max(0, min(len(times) - 2, np.searchsorted(times, t, side='right') - 1))
    if len(times) == 1:
        return anchors[0]
    if times[i + 1] - times[i] > .8:
        return anchors[nearest] if abs(times[nearest] - t) < .035 else None
    u = (t - times[i]) / max(1e-9, times[i + 1] - times[i])
    pos = (1 - u) * np.array(anchors[i]['position']) + u * np.array(anchors[i + 1]['position'])
    yaw0 = anchors[i]['yaw']; yaw1 = yaw0 + (anchors[i + 1]['yaw'] - yaw0 + 180) % 360 - 180
    return dict(anchors[nearest], position=pos.tolist(), yaw=float((1 - u) * yaw0 + u * yaw1))
