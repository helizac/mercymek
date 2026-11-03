"""Kinematics on synthetic tracks: constant velocity (speed within 2%), robust fit with outliers, gap geometry."""
import numpy as np

from mercy.analysis import physics as P


def test_constant_velocity_speed_within_2pct():
    t = np.arange(0, 6, 1 / 8)
    v = 15.0                                              # m/s along z
    centers = np.stack([np.zeros_like(t), np.full_like(t, 0.7), v * t], 1)
    speed, acc, vel, _ = P.kinematics(centers, t)
    assert np.all(np.abs(speed[2:-2] - v) / v < 0.02)
    assert np.all(np.abs(acc[3:-3]) < 0.5)


def test_robust_velocity_rejects_depth_spikes():
    t = np.arange(0, 1.5, 1 / 8)
    X = np.stack([3.0 * t, np.zeros_like(t), 10 + 8.0 * t], 1)
    X[4, 2] += 9.0; X[7, 2] -= 7.0                        # monocular depth outliers
    vel, _ = P.robust_velocity(t, X)
    assert np.allclose(vel, [3.0, 8.0], atol=0.4)


def test_gap_geometry_and_impact_frame():
    # two bodies on a collision course: A along +z at 15 m/s, B crossing along +x at 8 m/s; contact near t=2 s
    t = np.arange(0, 4, 1 / 8)
    A = np.stack([np.zeros_like(t), 15 * t], 1)
    B = np.stack([-16 + 8 * t, np.full_like(t, 32.0)], 1)
    gaps = [P.rect_rect_dist(a - [0.9, 2.25], a + [0.9, 2.25], b - [2.25, 0.9], b + [2.25, 0.9]) for a, b in zip(A, B)]
    k = int(np.argmin(gaps))
    assert abs(t[k] - 2.0) <= 0.25 and gaps[k] == 0.0
    assert P.rect_point_dist([0, 0], [2, 2], [3, 1]) == 1.0


def test_measurement_format():
    m = P.M("x.speed", "Speed", 50, 60, 40, "km/h", "test", "low")
    assert m["range"] == [40, 60] and m["unit"] == "km/h" and m["method"] and m["estimate"] is True


def test_named_collision_parties_do_not_include_witness_when_boxes_do_not_overlap():
    overlays={'0':{'car:rear':{'box':[100,200,150,250]},'truck:front':{'box':[160,200,220,260]}}}
    scene={'camera_vehicle_involved':False,'vehicles':[
        {'label':'dark hatchback','role':'striking'},{'label':'silver pickup','role':'struck'}]}
    involved,_=P.choose_objects({'car:rear':[],'truck:front':[]},overlays,
        {'width':1280,'height':720,'src_fps':30},scene,0.,
        {'car:rear':'dark hatchback','truck:front':'silver pickup'})
    assert set(involved)=={'car:rear','truck:front'}
