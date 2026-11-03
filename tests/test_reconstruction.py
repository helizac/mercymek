import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mercy.generation import cleanplate
from mercy.generation.vehicles import plate_for
from mercy.ingest.environment import fit_road, fit_observed_boxes, track_keys, transform_gaussians, write_splat
from mercy.ingest.source_quality import validate


def test_vehicle_registry_includes_distinct_round_and_offroad_bodies():
    from mercy.generation.vehicles import TYPES, DIMENSIONS, SCHEMA
    library=json.loads((Path(__file__).resolve().parents[1]/'data/models/vehicles/index.json').read_text())
    assert set(TYPES)=={m['type'] for m in library['models']}
    assert SCHEMA['properties']['body_type']['enum']==TYPES
    for kind in ['rounded_hatchback','offroad_suv']:
        model=next(m for m in library['models'] if m['type']==kind)
        assert model['dimensions_m']==DIMENSIONS[kind]
        assert (Path(__file__).resolve().parents[1]/'data/models/vehicles'/model['file']).stat().st_size>1000


def test_road_calibration_rejects_collapsed_geometry():
    with pytest.raises(ValueError):
        fit_road(np.zeros((1000, 3)), 1000, 1280, 720)


def test_road_calibration_recovers_tilt_and_camera_height():
    rng = np.random.default_rng(4)
    x = rng.uniform(-4, 4, 8000)
    z = rng.uniform(2, 8, 8000)
    P = np.column_stack([x, np.full_like(x, 1.35), z])
    tilt = Rotation.from_euler("z", 6, degrees=True).as_matrix()
    P = P @ tilt.T * 1.8
    road = fit_road(P, 900, 1280, 720)
    transformed = P @ np.array(road["basis"]).T * road["scale"] + [0, 1.35, 0]
    assert np.max(np.abs(transformed[:, 1])) < 1e-5
    assert road["scale"] == pytest.approx(1 / 1.8, rel=1e-4)


def test_gaussian_transform_preserves_covariance_and_file_layout(tmp_path):
    basis = np.diag([1., -1., 1.])
    g = {"pos": np.array([[2., 1.35, 5.]]), "scales": np.array([[.2, .4, .7]]),
         "quat": np.array([[1., 0., 0., 0.]])}
    pos, scales, quat = transform_gaussians(g, basis, 1)
    assert pos[0] == pytest.approx([2, 0, -5])
    R = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()[0]
    assert R @ np.diag(scales[0] ** 2) @ R.T == pytest.approx(np.diag([.04, .16, .49]))
    p = tmp_path / "world.splat"
    write_splat(p, pos, scales, quat, np.ones((1, 3)), np.ones(1))
    assert p.stat().st_size == 32
    assert np.frombuffer(p.read_bytes()[:12], "<f4") == pytest.approx([2, 0, -5])


def test_track_replay_preserves_full_turn_and_does_not_bridge_gaps():
    t = np.arange(0, 4, .125)
    rows = [{"t": float(x), "center": [float(4 * np.sin(x)), 0, float(4 * np.cos(x))]} for x in t]
    keys = track_keys(rows, lambda p: p, 2., 0, 4)
    assert len(keys) == len(rows)
    assert max(k[3] for k in keys) - min(k[3] for k in keys) > 90
    rows = [{"t": 0, "center": [0, 0, 0]}, {"t": .1, "center": [0, 0, .1]},
            {"t": 1.5, "center": [0, 0, .2]}, {"t": 1.6, "center": [0, 0, .3]}]
    keys = track_keys(rows, lambda p: p, 1.5, 0, 2)
    assert len(keys) == 2 and keys[0][0] == 0


def test_observed_boxes_refine_vehicle_registration():
    dims = [1.8, 1.5, 4.7]
    corners = np.array([[x,y,z] for x in [-.9,.9] for y in [0,1.5] for z in [-2.35,2.35]]) + [2,0,10]
    R = np.diag([1.,-1.,1.])
    C = np.array([0.,1.35,0.])
    pc = (corners-C) @ R
    uv = pc[:,:2]/pc[:,2:]*800+[640,360]
    box = [float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())]
    keys = [[t,3,15,0] for t in [0.,.125,.25]]
    M = np.eye(4); M[:3,:3]=R; M[:3,3]=C
    rec = {"times":np.array([0,.125,.25]),"intrinsic":np.repeat(np.array([[[160.,0,128],[0,160,72],[0,0,1]]]),3,axis=0),
           "depth":np.zeros((3,144,256))}
    overlays = {str(i):{"v":{"box":box}} for i in [0,1,2]}
    fitted, report = fit_observed_boxes(keys,"v",{"dimensions_m":dims},0,overlays,
        {"width":1280,"height":720,"src_fps":8},rec,[{"M":M}]*3,lambda p:p,np.eye(3))
    assert report["observations"] == 3
    assert fitted[1][1:3] == pytest.approx([2,10],abs=.25)


def test_plate_text_is_never_invented_or_transferred_between_vehicles():
    plates = {"plates": [{"key": "p1", "vehicle": "v1", "legible": True, "confidence": .9, "text": "AB?123"}]}
    assert plate_for("v1", plates)["text"] == "AB?123"
    assert plate_for("v2", plates)["text"] == ""
    assert plate_for("v1", dict(plates, blurred_at_source=True))["text"] == ""


def test_clean_image_review_is_invalidated_when_image_changes(tmp_path, monkeypatch):
    src, image = tmp_path / "source.png", tmp_path / "empty.png"
    src.write_bytes(b"source"); image.write_bytes(b"accepted")
    image.with_suffix(".review.json").write_text(json.dumps({"accepted": True,
        "source_sha256": cleanplate.digest(src), "image_sha256": cleanplate.digest(image)}))
    monkeypatch.setattr(cleanplate, "has_key", lambda: False)
    assert cleanplate.verified_clean_plate(src, image) == image
    image.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        cleanplate.verified_clean_plate(src, image)


def test_new_source_rejects_near_miss_blurred_plates_and_missing_license(tmp_path, monkeypatch):
    from mercy.ingest import source_quality
    monkeypatch.setattr(source_quality, "probe", lambda p: {"width": 1920, "height": 1080})
    video, manifest = tmp_path / "clip.mp4", tmp_path / "source.json"
    video.write_bytes(b"video")
    manifest.write_text(json.dumps({"review": {"event_kind": "near-miss", "plates_blurred": True}}))
    with pytest.raises(ValueError) as caught:
        validate(video, manifest)
    assert "open license" in str(caught.value) and "physical" in str(caught.value) and "unblurred" in str(caught.value)
