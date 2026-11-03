"""Continuous world motion: camera translation, bounded incident trajectories and source body heading.

This uses existing video-derived estimates and CPU geometry, not another neural inference. A detection
interval describes evidence, not physical existence. Unobserved continuation remains an explicit prior.
"""
import hashlib
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares

VERSION = 'continuous-world-motion-v2'


def curved_road(case):
    path=case/'scene.json'
    road=json.loads(path.read_text()).get('road_type','').lower() if path.exists() else ''
    return any(word in road for word in ['roundabout','curved','bend','winding','hairpin'])


def bounded_velocity(velocities, times, max_speed=40., max_acceleration=9.):
    result = np.asarray(velocities, float).copy()
    result *= np.minimum(1., max_speed / np.maximum(np.linalg.norm(result, axis=1), 1e-9))[:, None]
    for _ in range(4):
        for order in [range(1, len(result)), range(len(result)-2, -1, -1)]:
            for i in order:
                other = i-1 if order.step > 0 else i+1
                delta = result[i]-result[other]
                limit = max_acceleration * abs(times[i]-times[other])
                result[i] = result[other] + delta * min(1., limit/max(np.linalg.norm(delta), 1e-9))
    return result


def camera_path(case, times, matrices, anchor_t, return_velocity=False, write_report=True):
    """Do not interleave isolated PnP fixes with a zero/unrelated fallback camera translation."""
    physics = json.loads((case/'physics.json').read_text())
    result = np.array(matrices, copy=True)
    # Saved PnP cameras can predate the published road calibration. Their height must not
    # override the world the viewer actually loads: it biases every near-field box depth.
    road_path = case/'environment/reconstruction.json'
    road = json.loads(road_path.read_text()).get('road', {}) if road_path.exists() else {}
    height = float(road.get('camera_height_m', np.median(matrices[:, 1, 3])))
    if not np.isfinite(height) or height <= 0:
        raise ValueError('Invalid replay road camera height')
    result[:, 1, 3] = height
    series = physics.get('series', {}).get('objects', {}).get('ego')
    if not series or 'speed_kmh' not in series:
        return (result,None) if return_velocity else result
    times = np.asarray(times, float)
    dt = float(np.median(np.diff(times)))
    speed = np.interp(times, series['t'], series['speed_kmh']) / 3.6
    speed = gaussian_filter1d(np.clip(speed, 0, 40), max(.25/dt, .5), mode='nearest')
    lateral = gaussian_filter1d(median_filter(result[:, 0, 3], size=7, mode='nearest'), max(.3/dt, .5))
    vx = np.clip(np.gradient(lateral, times), -3, 3)
    curved=curved_road(case)
    if curved:
        forward=result[:,:3,2][:,[0,2]]
        angle=np.unwrap(np.arctan2(forward[:,0],forward[:,1]))
        angle=gaussian_filter1d(angle,max(.15/dt,.5),mode='nearest')
        velocity=speed[:,None]*np.column_stack([np.sin(angle),np.cos(angle)])
    else:
        velocity=np.column_stack([vx,speed])
    velocity = bounded_velocity(velocity, times, max_acceleration=8.)
    position = np.zeros((len(times), 2))
    position[1:] = np.cumsum((velocity[:-1]+velocity[1:])*.5*np.diff(times)[:, None], axis=0)
    origin = np.array([np.interp(anchor_t, times, matrices[:, j, 3]) for j in [0, 2]])
    position += origin - np.array([np.interp(anchor_t, times, position[:, j]) for j in range(2)])
    result[:, 0, 3] = position[:, 0]; result[:, 2, 3] = position[:, 1]
    report = dict(version=VERSION, source='existing source-video ego speed estimate; bounded integrated camera translation',
        assumption=True, measured_odometry=False, anchor_t=float(anchor_t),
        max_speed_mps=float(np.linalg.norm(velocity, axis=1).max()), max_acceleration_mps2=8.,
        camera_height_m=height, height_source='published road calibration' if 'camera_height_m' in road else 'camera pose prior',
        path_mode='source-camera heading on curved road' if curved else 'straight-road translation prior',
        raw_cameras_sha256=hashlib.sha256((case/'environment/cameras.npz').read_bytes()).hexdigest(),
        physics_sha256=hashlib.sha256((case/'physics.json').read_bytes()).hexdigest(),
        limitation='metric scale/speeds remain video-derived estimates; not independent sensor measurements')
    if write_report:
        (case/'vehicles/playback-camera.json').write_text(json.dumps(report, indent=2))
    full_velocity=np.column_stack([velocity[:,0],np.zeros(len(times)),velocity[:,1]])
    return (result,full_velocity) if return_velocity else result


def body_headings(case, item, times, cameras, fps):
    """A 2D bounding rectangle cannot determine whether the nose points left or right."""
    from mercy.ingest.vehicle_scene import heading
    path = case/'vehicles/heading-review.json'
    members = item['model'].get('source_keys', [item['key']])
    review = json.loads(path.read_text()) if path.exists() else {}
    samples = [h for h in review.get('headings', []) if h['track'] in members and h['confidence'] >= .65]
    if samples and not curved_road(case):
        samples.sort(key=lambda h:h['source_frame'])
        st = np.array([s['source_frame']/fps for s in samples])
        sy = np.unwrap(np.radians([s['road_yaw_deg'] for s in samples]))
        yaw = np.degrees(np.interp(times, st, sy))
        source = 'full-video road-relative body-heading review; image-side perspective distinguished from a turn'
    else:
        yaw = np.array([heading(item['model'], t*fps, fps, cameras.at(t)[1])[0] for t in times])
        yaw = np.degrees(np.unwrap(np.radians(yaw)))
        source = 'source appearance heading transformed by source-camera orientation; no per-frame box-shape angle optimization'
    return yaw, source


def fit_incident(times, boxes, valid, camera_values, dimensions, old_positions, sample_times, headings,
                 neural_anchors=(), max_speed=40., max_acceleration=10.):
    """One C1 trajectory, not independently fitted per-frame depths. Bound all velocity controls."""
    from mercy.ingest.road_motion import integrated_speed_basis
    from mercy.ingest.vehicle_scene import cuboid
    times = np.asarray(times, float); sample_times = np.asarray(sample_times, float)
    boxes = np.asarray(boxes, float); valid = np.asarray(valid, bool)
    C, R, K = map(np.asarray, zip(*camera_values))
    n = min(12, max(3, int(np.ceil((times[-1]-times[0])/.55))+1))
    knots = np.linspace(times[0], times[-1], n)
    B = integrated_speed_basis(times, knots)
    old = np.asarray(old_positions, float)
    # Median-filter depth spikes, especially at the first visible sliver of an occluded car.
    seed = median_filter(old, size=(min(9, len(old)//2*2+1), 1), mode='nearest')
    if len(neural_anchors) >= 2:
        at = np.array([a['t'] for a in neural_anchors]); ap = np.array([a['position'] for a in neural_anchors])
        seed = np.column_stack([np.interp(times, at, ap[:, j]) for j in range(2)])
    control_pos = np.column_stack([np.interp(knots, times, seed[:, j]) for j in range(2)])
    v0 = bounded_velocity(np.gradient(control_pos, knots, axis=0), knots, max_speed, max_acceleration)
    initial = np.r_[seed[0], v0.ravel()]
    corners = np.array([cuboid(dimensions, yaw) for yaw in headings])
    norms = np.column_stack([boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]]*2).clip(6)
    weights = np.sqrt(np.clip((boxes[:, 3]-boxes[:, 1])/40, .1, 1))[:, None]
    AB = integrated_speed_basis([a['t'] for a in neural_anchors], knots)
    AP = np.array([a['position'] for a in neural_anchors]).reshape(-1, 2)
    sigma = np.maximum(1.5, np.linalg.norm(AP-np.median(C[:, [0, 2]], axis=0), axis=1)*.12)

    def positions(params, basis):
        return params[:2]+basis@params[2:].reshape(n, 2)

    def project(P):
        pc = np.einsum('nki,nij->nkj', corners+np.column_stack([P[:, 0], np.zeros(len(P)), P[:, 1]])[:, None]-C[:, None], R)
        z = np.maximum(pc[:, :, 2], .2)
        u = K[:, 0, 0, None]*pc[:, :, 0]/z+K[:, 0, 2, None]
        v = K[:, 1, 1, None]*pc[:, :, 1]/z+K[:, 1, 2, None]
        return np.column_stack([u.min(1), v.min(1), u.max(1), v.max(1)])

    def residual(params):
        delta = (project(positions(params, B))-boxes)/norms
        edges = np.where(valid, delta, np.maximum(0, delta*[1, 1, -1, -1])*.2)*weights
        vel = params[2:].reshape(n, 2)
        acc = np.diff(vel, axis=0)/np.diff(knots)[:, None]
        anchors = ((positions(params, AB)-AP)/sigma[:, None]*.5).ravel()
        return np.r_[edges.ravel(), anchors, (acc*.045).ravel(),
            (np.maximum(0, np.linalg.norm(vel, axis=1)-max_speed)*.4),
            (np.maximum(0, np.linalg.norm(acc, axis=1)-max_acceleration)*.5),
            (positions(params, B)-seed).ravel()/500]

    lower = np.r_[[-300., -400.], np.full(n*2, -max_speed)]
    upper = np.r_[[300., 500.], np.full(n*2, max_speed)]
    initial = np.clip(initial, lower+1e-6, upper-1e-6)
    fitted = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1', f_scale=.12, max_nfev=120)
    params = fitted.x.copy()
    params[2:] = bounded_velocity(params[2:].reshape(n, 2), knots, max_speed, max_acceleration).ravel()
    # Once velocities are feasible, refit just the translation, retaining the hard motion bounds.
    offset = least_squares(lambda p:residual(np.r_[p, params[2:]]), params[:2], loss='soft_l1', f_scale=.12, max_nfev=35)
    params[:2] = offset.x
    E = integrated_speed_basis(np.clip(sample_times, times[0], times[-1]), knots)
    P = positions(params, E)
    vel = params[2:].reshape(n, 2)
    P += np.minimum(0, sample_times-times[0])[:, None]*vel[0]
    P += np.maximum(0, sample_times-times[-1])[:, None]*vel[-1]
    yaw = np.interp(sample_times, times, np.unwrap(np.radians(headings)))
    error = abs(project(positions(params, B))-boxes)[valid]
    info = dict(model='joint incident path with continuous bounded velocity/acceleration; source body direction',
        straight_prior_applied=False, max_speed_mps=float(np.linalg.norm(vel, axis=1).max()),
        max_acceleration_mps2=float(np.linalg.norm(np.diff(vel, axis=0)/np.diff(knots)[:, None], axis=1).max()),
        speed_limit_mps=max_speed, acceleration_limit_mps2=max_acceleration,
        neural_anchor_count=len(neural_anchors), median_valid_edge_error_px=float(np.median(error)) if len(error) else None,
        assumption=True, outside_fit_window='bounded boundary-velocity continuation; actor remains in world')
    return np.column_stack([P, np.degrees(yaw)]), info
